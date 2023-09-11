
# TensorFlow 2 Object Detection API tutorial

<https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html>

# This repository must be cloned to the main project folder

<https://github.com/tensorflow/models.git>

# Download and unzip pre-treined-model into ./workspace/training_demo/pre-trained-models

<http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz>

# Model training results

| Date       | Model name                          | Steps | Changed parameters                                                                                                                                                                                                            | Total lose                                                                                                                                                                                       | Training time |
|------------|-------------------------------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| 08.09.2023 | my_ssd_resnet50_v1_fpn_3(attempt 1) | 2000  | box_coder<br/> faster_rcnn_box_coder <br/>y_scale: 32.0 <br/> x_scale: 32.0 <br/> height_scale: 3.0<br/> width_scale: 3.0                                                                                                     |Loss/classification_loss': 0.087042436,<br/> 'Loss/localization_loss': 0.1134364, <br/>'Loss/regularization_loss': 141.97772,<br/> 'Loss/total_loss': 142.17819,<br/> 'learning_rate': 0.02765367 |               |
| 08.09.2023 | my_ssd_resnet50_v1_fpn_3(attempt 2) | 2000  | box_coder<br/> faster_rcnn_box_coder <br/>y_scale: 32.0 <br/> x_scale: 32.0 <br/> height_scale: 3.0<br/> width_scale: 3.0 <br/> feature_extractor <br/> regularizer  <br/> l2_regularizer <br/> weight: 0.0039999998989515007 |Loss/classification_loss': 0.08048342,<br/> 'Loss/localization_loss': 0.08371392, <br/>  'Loss/regularization_loss': 1.8251712,<br/>  'Loss/total_loss': 1.9893686, <br/> 'learning_rate': 0.02765367} |               |
| 08.09.2023 | my_ssd_resnet50_v1_fpn_3(attempt 2) | 5000  | box_coder<br/> faster_rcnn_box_coder <br/>y_scale: 32.0 <br/> x_scale: 32.0 <br/> height_scale: 3.0<br/> width_scale: 3.0 <br/> feature_extractor <br/> regularizer  <br/> l2_regularizer <br/> weight: 0.0039999998989515007 |Loss/classification_loss': 0.04645691,<br/> 'Loss/localization_loss': 0.03727855,<br/> 'Loss/regularization_loss': 1.218591,<br/> 'Loss/total_loss': 1.3023264,<br/> 'learning_rate': 0.0|               |
| 08.09.2023 | my_ssd_resnet50_v1_fpn_4(attempt 1) | 2000  | default settings                                                                                                                                                                                                              | 'Loss/classification_loss': 0.14104246,<br/> 'Loss/localization_loss': 0.036720034,<br/>'Loss/regularization_loss': 0.5548869,<br/> 'Loss/total_loss': 0.7326494,<br/> 'learning_rate': 0.02765367}                |
                           

                                                                                                
