import os
import xml.etree.ElementTree as ET
from shutil import copyfile


signs =   { 0:'Speed limit (50km/h)',            
            1:'No vehicles',  
            2:'Veh > 3.5 tons prohibited', 
            3:'No entry',          
            4:'Turn right ahead',
            5:'Turn left ahead', 
            6:'Ahead only', 
            7:'Go straight or right', 
            8:'Go straight or left',           
            9:'Speed limit (90km/h)' }



XML_FILE_PATH = "C:\\Projects\\TFOD\\workspace\\training_demo\\new_image"
COPY_TO_PATH = "C:\\Projects\\TFOD\\workspace\\training_demo\\10_class_images\\new"

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    label_list = []
    

    for obj in root.findall('object'):
        label = obj.find('name').text    

        label_list.append(label)        

    return label_list[0]

def copy_files(xml_file, destination):
     filename = os.path.basename(xml_file)
     copyfile(xml_file, os.path.join(destination, filename))
     jpg_file = os.path.splitext(xml_file)[0]+'.jpg'
     if os.path.exists(jpg_file):
        filename = os.path.basename(jpg_file)
        copyfile(jpg_file, os.path.join(destination, filename))
     else:
          print(f"There isn't {jpg_file}!")


     

for root, dir, files in os.walk(XML_FILE_PATH):
    for file in files:
        if file.lower().endswith(('xml')):
                class_name = parse_xml(os.path.join(root, file))
                for value in signs.values():
                    if value  ==  class_name :
                        copy_files(os.path.join(root, file), COPY_TO_PATH)
                