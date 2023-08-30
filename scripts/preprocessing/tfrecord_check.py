import tensorflow as tf
import matplotlib.pyplot as plt

# Define the feature description for parsing the TFRecord
feature_description = {
    'width': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'x_min': tf.io.FixedLenFeature([], tf.float32),
    'y_min': tf.io.FixedLenFeature([], tf.float32),
    'x_max': tf.io.FixedLenFeature([], tf.float32),
    'y_max': tf.io.FixedLenFeature([], tf.float32),
    'class_id': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}

# Function to parse a TFRecord example
def parse_tfrecord_fn(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example

# Path to your TFRecord file
tfrecord_file = 'workspace/training_demo/annotations/test.record'

# Define a mapping from class IDs to class names
class_id_to_name = {
    0: 'ClassA',
    1: 'ClassB',
    # ... and so on
}

# Create a TFRecordDataset from the file
dataset = tf.data.TFRecordDataset(tfrecord_file)

# Parse the records using the parsing function
parsed_dataset = dataset.map(parse_tfrecord_fn)

# Iterate through the parsed records and display the images
for record in parsed_dataset.take(5):  # Display images from the first 5 records
    image_bytes = record['image'].numpy()
    image = tf.image.decode_jpeg(image_bytes, channels=3)  # Decode JPEG image
    class_id = record['class_id'].numpy()

    class_name = class_id_to_name.get(class_id, f'Unknown Class {class_id}')

    plt.figure()
    plt.imshow(image)
    plt.title(f"Class Name: {class_name}")
    plt.axis('off')
    plt.show()