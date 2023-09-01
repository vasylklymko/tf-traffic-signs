import os
import numpy as np
import cv2
from lxml import etree

# Input and output directories
input_image_dir = "workspace/training_demo/images/no_entry/"
input_annotation_dir = "workspace/training_demo/images/no_entry/"
output_image_dir = "workspace/training_demo/images/no_entry_augmented/"
output_annotation_dir = "workspace/training_demo/images/no_entry_augmented/"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

# Augmentation parameters
num_augmentations = 5  # Number of augmentations per image
augmentation_factor = 1.2  # Scale factor for resizing
rotation_angles = [-2, 0, 2]  # Rotation angles in degrees
brightness_factors = [0.7, 1.0, 1.3]  # Brightness factors

# Function to parse an XML annotation file
def parse_annotation(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objects.append((label, (xmin, ymin, xmax, ymax)))
    return filename, objects

# Function to generate augmented images and annotations
def augment_image_and_annotation(image_path, annotation_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Parse the original annotation
    filename, objects = parse_annotation(annotation_path)

    for i in range(num_augmentations):
        # Create a copy of the original image and annotation
        augmented_image = image.copy()
        augmented_objects = objects.copy()

        # Apply random augmentation transformations
        scale_factor = np.random.uniform(1, augmentation_factor)
        angle = np.random.choice(rotation_angles)
        brightness_factor = np.random.choice(brightness_factors)

        # Rescale
        augmented_image = cv2.resize(
            augmented_image, (int(width * scale_factor), int(height * scale_factor))
        )

        # Rotate
        rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (width, height))

        # Adjust brightness
        augmented_image = cv2.convertScaleAbs(augmented_image, alpha=brightness_factor)

        # Update bounding box coordinates
        for j in range(len(augmented_objects)):
            _, bbox = augmented_objects[j]
            xmin, ymin, xmax, ymax = bbox
            bbox = (
                int(xmin * scale_factor),
                int(ymin * scale_factor),
                int(xmax * scale_factor),
                int(ymax * scale_factor),
            )
            augmented_objects[j] = (_, bbox)

        # Save augmented image
        output_image_path = os.path.join(output_dir, f"{filename.split('.')[0]}_{i}.jpg")
        cv2.imwrite(output_image_path, augmented_image)

        # Generate and save augmented annotation XML
        output_xml_path = os.path.join(output_dir, f"{filename.split('.')[0]}_{i}.xml")
        generate_annotation_xml(filename, augmented_objects, output_xml_path)

# Function to generate and save annotation XML
def generate_annotation_xml(filename, objects, output_xml_path):
    root = etree.Element("annotation")
    filename_elem = etree.Element("filename")
    filename_elem.text = filename
    root.append(filename_elem)

    for obj in objects:
        label, bbox = obj
        object_elem = etree.Element("object")
        name_elem = etree.Element("name")
        name_elem.text = label
        object_elem.append(name_elem)

        bbox_elem = etree.Element("bndbox")
        xmin_elem = etree.Element("xmin")
        ymin_elem = etree.Element("ymin")
        xmax_elem = etree.Element("xmax")
        ymax_elem = etree.Element("ymax")

        xmin_elem.text = str(bbox[0])
        ymin_elem.text = str(bbox[1])
        xmax_elem.text = str(bbox[2])
        ymax_elem.text = str(bbox[3])

        bbox_elem.append(xmin_elem)
        bbox_elem.append(ymin_elem)
        bbox_elem.append(xmax_elem)
        bbox_elem.append(ymax_elem)

        object_elem.append(bbox_elem)
        root.append(object_elem)

    xml_str = etree.tostring(root, pretty_print=True).decode("utf-8")
    with open(output_xml_path, "w") as xml_file:
        xml_file.write(xml_str)

# Process each image and its annotation
for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_image_dir, filename)
        annotation_path = os.path.join(input_annotation_dir, f"{filename.split('.')[0]}.xml")
        augment_image_and_annotation(image_path, annotation_path, output_annotation_dir)

print("Augmentation and annotation generation completed.")
