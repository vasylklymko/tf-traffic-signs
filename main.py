import os  # Suppress TensorFlow logging
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

PATH_TO_SAVED_MODEL = "workspace/training_demo/exported-models/my_model" + "/saved_model"
PATH_TO_LABELS = "workspace/training_demo/annotations/label_map.pbtxt"
PATH_TO_TESTING_IMAGES = "workspace/training_demo/images/testing"
PATH_TO_CFG = 'workspace/training_demo/exported-models/my_model/pipeline.config'
PATH_TO_CKPT = 'workspace/training_demo/workspace/training_demo/exported-models/my_model/checkpoint'

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

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

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (80, 80))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)  # Convert to uint8
    input_tensor = tf.expand_dims(input_tensor, 0)

    # Perform inference
    detections = model(input_tensor)

    # Process and visualize detections
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    for i, score in enumerate(scores):
        if score > 0.50:  # Filter out low-confidence detections
            ymin, xmin, ymax, xmax = boxes[i]
            left = int(xmin * frame.shape[1])
            top = int(ymin * frame.shape[0])
            right = int(xmax * frame.shape[1])
            bottom = int(ymax * frame.shape[0])

            class_id = classes[i]
            class_name = category_index[class_id]['name']
            label = f"{class_name}: {score:.2f}"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

