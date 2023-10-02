import os  # Suppress TensorFlow logging
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pandas as pd
import detect_time
from test_report.report import Report

# Classes of traffic signs
signs =   { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

PATH_TO_SAVED_MODEL = "workspace/training_demo/exported-models/my_model_22" + "/saved_model"
PATH_TO_LABELS = "workspace/training_demo/annotations/label_map.pbtxt"
PATH_TO_TESTING_IMAGES = "workspace/training_demo/new_image/test"
#PATH_TO_CFG = 'workspace/training_demo/exported-models/my_model/pipeline.config'
#PATH_TO_CKPT = 'workspace/training_demo/workspace/training_demo/exported-models/my_model/checkpoint'
PATH_TO_REPORT = 'workspace/training_demo/reports'

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
        if image.lower().endswith(('jpeg', 'jpg', 'png')):
          image_path.append(os.path.join( testing_folder,image))    

    return image_path

IMAGE_PATHS = get_testing_images(PATH_TO_TESTING_IMAGES)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2 
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

# Init the test report
report = Report("Model_19", PATH_TO_TESTING_IMAGES, PATH_TO_REPORT)


for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = cv2.imread(image_path)
    start_time = time.time()
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

    elapsed_time = time.time() - start_time
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
          agnostic_mode=False)
    
      
    # Define the desired window size (larger than the image)
    window_width = 800
    window_height = 600

    # Create a window with the specified size
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', window_width, window_height)

    # Calculate the position to center the image in the window
    x_position = (window_width - image_np_with_detections.shape[1]) // 2
    y_position = (window_height - image_np_with_detections.shape[0]) // 2

    # Create an empty canvas with the specified window size
    canvas = 255 * np.ones((window_height, window_width, 3), dtype=np.uint8)

    # Paste the small image onto the canvas at the calculated position
    canvas[y_position:y_position + image_np_with_detections.shape[0], x_position:x_position + image_np_with_detections.shape[1]] = image_np_with_detections
    canvas = detect_time.display_detect_time(canvas, elapsed_time)
    cv2.imshow('Detection', canvas)

    
    detection_table = pd.DataFrame(zip(detections['detection_classes'], detections['detection_scores']))
    label_table = pd.DataFrame(signs.items())
    label_table.iloc[:,0]+=1
    table = detection_table.merge(label_table, on=0, how='left')

    report.append_results(image_path, table.iloc[0, 2], detections['detection_scores'][0])

    print(table[:5])
    print('Done')
    time.sleep(0.5)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

report.__del__()

   

