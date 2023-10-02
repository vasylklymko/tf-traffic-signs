import os

def rename_images_in_folder(folder_path, prefix="roundabout_mandatory_image_", start_number=1):
    try:
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]

        if not image_files:
            print(f"No image files found in '{folder_path}'.")
            return

        for index, filename in enumerate(image_files, start=start_number):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}{index:03d}{file_extension}"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

        print("Renaming completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    folder_path = "workspace\\training_demo\\new_image\\roundabout_mandatory" 
    rename_images_in_folder(folder_path)
