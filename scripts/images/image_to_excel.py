import os
import xlsxwriter

# Create a new Excel workbook
workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()

# Specify the folder where your images are located
image_folder = 'C:\\Users\\vasyl.klymko\\Downloads\\Road-Sing_DataSet\\Meta'

# Specify the starting cell for inserting images (e.g., A1)
start_row = 1
start_col = 1

# Set the width and height for the cells to accommodate the images
cell_width = 100 # Adjust this value based on your desired cell width
cell_height = 100  # Adjust this value based on your desired cell height

# Iterate through the files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Filter by image file extensions
        # Add the image as a shape bound to the cell
        img_path = os.path.join(image_folder, filename)
        worksheet.insert_image(start_row, start_col, img_path, {'x_scale': cell_width/100, 'y_scale': cell_height/100})

        # Move to the next row for the next image
        start_row += 1

# Close the workbook
workbook.close()
