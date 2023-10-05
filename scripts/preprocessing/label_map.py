
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
import os

# Classes of traffic signs
# signs =   { 0:'Speed limit (20km/h)',
#             1:'Speed limit (30km/h)', 
#             2:'Speed limit (50km/h)', 
#             3:'Speed limit (60km/h)', 
#             4:'Speed limit (70km/h)', 
#             5:'Speed limit (80km/h)', 
#             6:'End of speed limit (80km/h)', 
#             7:'Speed limit (100km/h)', 
#             8:'Speed limit (120km/h)', 
#             9:'No passing', 
#             10:'No passing veh over 3.5 tons', 
#             11:'Right-of-way at intersection', 
#             12:'Priority road', 
#             13:'Yield', 
#             14:'Stop', 
#             15:'No vehicles', 
#             16:'Veh > 3.5 tons prohibited', 
#             17:'No entry', 
#             18:'General caution', 
#             19:'Dangerous curve left', 
#             20:'Dangerous curve right', 
#             21:'Double curve', 
#             22:'Bumpy road', 
#             23:'Slippery road', 
#             24:'Road narrows on the right', 
#             25:'Road work', 
#             26:'Traffic signals', 
#             27:'Pedestrians', 
#             28:'Children crossing', 
#             29:'Bicycles crossing', 
#             30:'Beware of ice/snow',
#             31:'Wild animals crossing', 
#             32:'End speed + passing limits', 
#             33:'Turn right ahead', 
#             34:'Turn left ahead', 
#             35:'Ahead only', 
#             36:'Go straight or right', 
#             37:'Go straight or left', 
#             38:'Keep right', 
#             39:'Keep left', 
#             40:'Roundabout mandatory', 
#             41:'End of no passing', 
#             42:'End no passing veh > 3.5 tons' }

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

def create_label_map(items):
    label_map = StringIntLabelMap()

    for item in items:
        
        label_map_item = StringIntLabelMapItem()
        label_map_item.id = item + 1
        label_map_item.name = items[item]        
        label_map.item.append(label_map_item)

    return label_map

def main():
    label_map = create_label_map(signs)  
    label_map_text = str(label_map)
    annotation_folder = './workspace/training_demo/annotations'
    file_name = 'label_map.pbtxt'
    file_path = os.path.join(annotation_folder, file_name)    
    with open(file_path, 'w') as label_map_file:
        label_map_file.write(label_map_text)

if __name__ == '__main__':
    main()