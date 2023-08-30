import csv
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_xml_file(image_info, image_folder):
    width, height, x1, y1, x2, y2, class_id, image_path = image_info
    image_name = os.path.basename(image_path)
    
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.dirname(image_path)
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name
    
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    
    object_elem = ET.SubElement(annotation, "object")
    name = ET.SubElement(object_elem, "name")
    name.text = str(class_id)
    bndbox = ET.SubElement(object_elem, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(x1)
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(y1)
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(x2)
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(y2)
    
    xml_string = ET.tostring(annotation, "utf-8")
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    xml_filename = os.path.splitext(image_name)[0] + ".xml"
    xml_path = os.path.join(image_folder,  folder.text, xml_filename)
    
    with open(xml_path, "w") as xml_file:
        xml_file.write(pretty_xml)

def main():
    csv_file_path = "workspace/training_demo/convert_images/test_1.csv"  # Update this with your CSV file path
    image_folder = os.path.dirname(csv_file_path)
    print (image_folder)
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            image_info = [int(val) if val.isdigit() else val for val in row]
            create_xml_file(image_info, image_folder)
    
    print("XML files created successfully.")

if __name__ == "__main__":
    main()
