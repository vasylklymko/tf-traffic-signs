import tensorflow as tf
import pandas as pd
import os
from PIL import Image

# Define feature description
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

# Function to serialize example
def serialize_example(width, height, x_min, y_min, x_max, y_max, class_id, image):
    feature = {
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'x_min': tf.train.Feature(float_list=tf.train.FloatList(value=[x_min])),
        'y_min': tf.train.Feature(float_list=tf.train.FloatList(value=[y_min])),
        'x_max': tf.train.Feature(float_list=tf.train.FloatList(value=[x_max])),
        'y_max': tf.train.Feature(float_list=tf.train.FloatList(value=[y_max])),
        'class_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Function to read and preprocess image
def read_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Assuming PNG images with 3 channels
    image = tf.image.resize(image, [32, 32])  # Adjust to your desired size
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return image


def main():
# Path to your CSV file
    csv_file = 'workspace/training_demo/images/train_1.csv'
    tfrecord_file = 'workspace/training_demo/annotations/train.record'

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        df = pd.read_csv(csv_file)
        for index, row in df.iterrows():
            image_dir = './workspace/training_demo/images'
            image_path = os.path.join(image_dir, row['Path'])
            print(image_path)
            image = read_and_preprocess_image(image_path)
            image_bytes = tf.io.encode_png(image).numpy()  # Convert image to bytes
            
            example = serialize_example(
                row['Width'],
                row['Height'],
                row['Roi.X1'] / row['Width'],    # Normalize bbox coordinates
                row['Roi.Y1'] / row['Height'],
                row['Roi.X2'] / row['Width'],
                row['Roi.Y2'] / row['Height'],
                row['ClassId'],
                image_bytes
            )
            writer.write(example)

    print(f"Conversion complete. TFRecord saved to {tfrecord_file}")

if __name__ == '__main__':
    main()

