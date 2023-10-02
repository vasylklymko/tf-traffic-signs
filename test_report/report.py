
import pandas as pd
import os
from . import label_reader
from datetime import datetime

class Report:    

    def __init__(self, model_name, test_dataset_path, report_path):
        self.detection_results = []
        self.model_name = model_name
        self.test_dataset = test_dataset_path
        self.report_path = report_path
        self.image_counter = 0
        self.true_counter = 0
        

    def append_results(self, file_name, detected_class, probability):
         # Increment the image counter
         self.image_counter+=1

         # Read the labeled class of image
         labeled_class = label_reader.read_label(file_name)

         # Compare detected class with labeled class
         classification = labeled_class == detected_class

        # Count the true classification case
         if classification:
             self.true_counter+=1


         # Create a dictionary for the detection result
         result = {            
            'File Name': file_name,
            'Class Label': labeled_class,
            'Detected Class': detected_class,
            'Probability': probability ,
            'Classification' : classification
            }
             
         # Append the result to the list of detection results
         self.detection_results.append(result)
         
         # Save the report file every 10 image classification
         if self.image_counter % 10 == 0:
             self.write() 
    
    def write(self):
        output_file_name = f"{self.model_name}-{datetime.now().strftime('%Y-%m-%d')}.csv"
        output_file = os.path.join(self.report_path,output_file_name )

        result = {            
            'File Name': "",
            'Class Label': "",
            'Detected Class': "",
            'Probability': "Images/classified" ,
            'Classification' : f"{self.image_counter}/{self.true_counter}"
            }

        self.detection_results.insert(0, result)

        df = pd.DataFrame(self.detection_results)        
        if not os.path.exists(output_file):
            df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, mode='w', header=False, index=False)

    def __del__(self):
         self.write()    
        
