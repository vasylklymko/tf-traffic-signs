
# TensorFlow 2 Object Detection API tutorial

<https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html>

# This repository must be cloned to the main project folder

<https://github.com/tensorflow/models.git>

# Download and unzip pre-treined-model into ./workspace/training_demo/pre-trained-models

<http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>

# Model training results

| Date          | Model name               | Steps | Changed parameters                                                                                   |  Total lose | Training time |
|---------------|--------------------------|-------|------------------------------------------------------------------------------------------------------|-------------|---------------|
| 06.09.2023    | my_ssd_resnet50_v1_fpn_3 | 5000  | box_coder<br/> faster_rcnn_box_coder <br/>y_scale: 32.0 <br/> x_scale: 32.0 <br/> height_scale: 3.0<br/> width_scale: 3.0 |             |               |
                                                                                                
