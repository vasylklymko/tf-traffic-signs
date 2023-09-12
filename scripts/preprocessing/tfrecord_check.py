import tensorflow as tf
import matplotlib.pyplot as plt

# Define the feature description for parsing the TFRecord
feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

# Function to parse a TFRecord example
def parse_tfrecord_fn(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example

# Path to your TFRecord file
tfrecord_file = 'E:\\git\\tf-traffic-signs\\workspace\\training_demo\\annotations\\train.record'

# Create a TFRecordDataset from the file
dataset = tf.data.TFRecordDataset(tfrecord_file)

# Parse the records using the parsing function
parsed_dataset = dataset.map(parse_tfrecord_fn)

# Iterate through the parsed records and inspect the data
for record in parsed_dataset.take(30):  # Inspect the first 5 records
    # Access features based on the provided schema
    height = record['image/height'].numpy()
    width = record['image/width'].numpy()
    filename = record['image/filename'].numpy()
    source_id = record['image/source_id'].numpy()
    image_encoded = record['image/encoded'].numpy()
    image_format = record['image/format'].numpy()
    
    # Access class-related features
    class_names = record['image/object/class/text'].values.numpy()
    class_labels = record['image/object/class/label'].values.numpy()

    # Access bounding box coordinates
    xmins = record['image/object/bbox/xmin'].values.numpy()
    xmaxs = record['image/object/bbox/xmax'].values.numpy()
    ymins = record['image/object/bbox/ymin'].values.numpy()
    ymaxs = record['image/object/bbox/ymax'].values.numpy()

    print("Height:", height)
    print("Width:", width)
    print("Filename:", filename)
    print("Source ID:", source_id)
    print("Image Format:", image_format)
    print("Class Names:", class_names)
    print("Class Labels:", class_labels)
    print("Bounding Box Xmins:", xmins)
    print("Bounding Box Xmaxs:", xmaxs)
    print("Bounding Box Ymins:", ymins)
    print("Bounding Box Ymaxs:", ymaxs)

    image_encoded = record['image/encoded'].numpy()
    image_format = record['image/format'].numpy()

    # Decode the image based on the format (JPEG or PNG)
    if image_format == b'jpg':
        image = tf.image.decode_jpeg(image_encoded, channels=3)
    elif image_format == b'png':
        image = tf.image.decode_png(image_encoded, channels=3)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")

    # Display the image
    plt.figure()
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')
    plt.show()    
