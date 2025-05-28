import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def merge_folders(folder1, folder2, destination_folder):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(folder1):
        source_path = os.path.join(folder1, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)

    for filename in os.listdir(folder2):
        source_path = os.path.join(folder2, filename)
        destination_path = os.path.join(destination_folder, filename)

        if os.path.exists(destination_path):
            base, extension = os.path.splitext(filename)
            new_filename = f"{base}_copy{extension}"
            destination_path = os.path.join(destination_folder, new_filename)

        shutil.copy2(source_path, destination_path)

folder1 = './Dataset/HAM10000_images_part_1'
folder2 = './Dataset/HAM10000_images_part_2'
destination_folder = './Dataset/train_image'

merge_folders(folder1, folder2, destination_folder)

print("Images merged successfully!")


def organize_dataset(image_folder: str, csv_file: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):

    metadata = pd.read_csv(csv_file)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = metadata['dx'].unique()

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    def find_image_file(image_id):
        image_file = image_id + '.jpg'
        full_path = os.path.join(image_folder, image_file)
        return full_path if os.path.exists(full_path) else None

    for class_name in classes:
        class_images = metadata[metadata['dx'] == class_name]['image_id'].tolist()
        train_images, test_images = train_test_split(class_images, test_size=test_size, random_state=random_state)

        for image_id in train_images:
            source_file = find_image_file(image_id)
            if source_file:
                shutil.copy(source_file, os.path.join(train_dir, class_name, os.path.basename(source_file)))

        for image_id in test_images:
            source_file = find_image_file(image_id)
            if source_file:
                shutil.copy(source_file, os.path.join(test_dir, class_name, os.path.basename(source_file)))

    print(f"Dataset organized into '{train_dir}' and '{test_dir}' subfolders successfully!")

    organize_dataset(
    image_folder='./Dataset/train_image',
    csv_file='./Dataset/HAM10000_metadata.csv',
    output_dir='./Dataset',
    test_size=0.2,
    random_state=42
)

