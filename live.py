import os  # Suppress TensorFlow logging
import time
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import detect_time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

PATH_TO_SAVED_MODEL = "workspace/training_demo/exported-models/my_model" + "/saved_model"
PATH_TO_LABELS = "workspace/training_demo/annotations/label_map.pbtxt"
PATH_TO_TESTING_IMAGES = "workspace/training_demo/images/testing"
PATH_TO_CFG = 'workspace/training_demo/models/my_ssd_resnet50_v1_fpn_22/pipeline.config'
PATH_TO_CKPT = 'workspace/training_demo/models/my_ssd_resnet50_v1_fpn_22/'

print('Loading model...', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-301')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

### Load label map data ###
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,  use_display_name=True)




cap = cv2.VideoCapture(0)

frame_counter = 0
start_time= time.time()

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    if not ret:
        break

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=5,
          min_score_thresh=.50,
          agnostic_mode=False)
    
    frame_counter+=1
    elapsed_time = time.time() - start_time
    fps = frame_counter / elapsed_time

    image_np_with_detections = detect_time.display_detect_time(image_np_with_detections, fps, "FPS")

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

