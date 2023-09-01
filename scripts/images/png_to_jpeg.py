from PIL import Image
import os
import pandas as pd
import cv2


# Convert PNG to JPEG
def convert_png_to_jpeg(input_path, output_path, quality=95):
    
    try:
        image = cv2.imread(input_path)
        output_path = output_path.replace(".png", ".jpg")
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        print(f"Conversion successful: {input_path} -> {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Provide paths to the input PNG file and the output JPEG file
csv_file = 'workspace/training_demo/images/test_1.csv'  
image_dir = './workspace/training_demo/images'
convert_image_dir = './workspace/training_demo/convert_images/' 

df = pd.read_csv(csv_file)

for index, row in df.iterrows():
    convert_image_dir = './workspace/training_demo/images'
    convert_image_dir = './workspace/training_demo/convert_images/'
    input_png_path = os.path.join(image_dir, row['Path'])  

    
    output_jpeg_path  =  os.path.join(convert_image_dir, row['Path'])    
    print(input_png_path)
    # Convert PNG to JPEG
    convert_png_to_jpeg(input_png_path, output_jpeg_path, quality=100)