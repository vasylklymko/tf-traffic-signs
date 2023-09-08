import csv
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

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


def create_xml_file(image_info, image_folder):
    width, height, x1, y1, x2, y2, class_id, image_path = image_info
    image_name = os.path.basename(image_path)
    
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.dirname(image_path)

    # for the train folder

    fold, _ = os.path.split(os.path.dirname(image_path))        
    if str(fold) == "train":
        folder.text = str(fold)
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name.replace(".png", ".jpg")
    
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    
    object_elem = ET.SubElement(annotation, "object")
    name = ET.SubElement(object_elem, "name")

    name.text = str(signs[class_id])
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
    xml_path = os.path.join(image_folder, folder.text, xml_filename)
    
    with open(xml_path, "w") as xml_file:
        xml_file.write(pretty_xml)

def main():
    csv_file_path = "workspace/training_demo/43_convert_images/train.csv"  # Update this with your CSV file path
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
