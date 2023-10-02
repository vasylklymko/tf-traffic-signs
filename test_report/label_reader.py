import os
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    label_list = []
    

    for obj in root.findall('object'):
        label = obj.find('name').text     

        label_list.append(label)
        

    return label_list[0]


def read_label(file_name):
    class_name = 'None'

    # Check if the file has a jpg, jpeg, or png extension
    if str(file_name).lower().endswith(('jpeg', 'jpg', 'png')):
       xml_file = os.path.splitext(file_name)[0] +'.xml' 
       if os.path.exists(xml_file):
            class_name = parse_xml(xml_file)
       else:
            print("There isn't any XML file.")           

    return class_name

