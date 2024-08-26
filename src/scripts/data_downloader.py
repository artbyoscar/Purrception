import os
import gdown
import zipfile

def download_and_extract(file_id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'dataset.zip')
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
    
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.remove(output_path)

# Download and extract datasets
datasets = [
    ('1nxOWNJwSEMhhh0KrPy2o9LuICza2miKr', 'catmeows_dataset1'),
    ('1M-PH4szGZd7N9qN26dz1CgqUqGcZ08vG', 'catmeows_dataset2'),
    ('1tYuzkFVJ9FYUE3UmOt7MVdQ_DLCuHejp', 'catmeows_dataset3')
]

for file_id, folder_name in datasets:
    download_and_extract(file_id, os.path.join('data', folder_name))

print("Datasets downloaded and extracted successfully.")