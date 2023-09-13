import os  # Suppress TensorFlow logging
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

PATH_TO_SAVED_MODEL = "workspace/training_demo/exported-models/my_model_7" + "/saved_model"
PATH_TO_LABELS = "workspace/training_demo/annotations/label_map.pbtxt"
PATH_TO_TESTING_IMAGES = "workspace/training_demo/images/testing"
#PATH_TO_CFG = 'workspace/training_demo/exported-models/my_model/pipeline.config'
#PATH_TO_CKPT = 'workspace/training_demo/workspace/training_demo/exported-models/my_model/checkpoint'

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

### Load label map data ###
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,  use_display_name=True)

def get_testing_images(testing_folder):
    image_path =[]
    for image in os.listdir(testing_folder):
        image_path.append(os.path.join( testing_folder,image))    

    return image_path

IMAGE_PATHS = get_testing_images(PATH_TO_TESTING_IMAGES)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2 
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = cv2.imread(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    #image_np = np.tile(
       #np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          line_thickness = 2,
          use_normalized_coordinates=True,
          max_boxes_to_draw=10,
          min_score_thresh=.5,
          agnostic_mode= False)
    
    
    time.sleep(3)
    cv2.imshow('Detection', image_np_with_detections)
    print('Done')
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



   

