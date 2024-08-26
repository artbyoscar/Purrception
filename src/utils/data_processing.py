import json
import os
import random
import shutil
from tqdm import tqdm

def process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Load the COCO format annotations for keypoints
    with open(os.path.join(keypoint_path, 'annotations.json'), 'r') as f:
        keypoint_data = json.load(f)

    # Load the COCO format annotations for bounding boxes
    boundingbox_files = [f for f in os.listdir(os.path.join(boundingbox_path, 'bndbox_anno')) if f.endswith('.json')]
    boundingbox_data = []
    for file in boundingbox_files:
        with open(os.path.join(boundingbox_path, 'bndbox_anno', file), 'r') as f:
            data = json.load(f)
            print(f"Structure of {file}:")
            print(json.dumps(data, indent=2)[:500])  # Print first 500 characters of the structure
            if 'annotations' in data:
                boundingbox_data.extend(data['annotations'])
            elif isinstance(data, list):
                boundingbox_data.extend(data)
            else:
                print(f"Unexpected structure in {file}")

    # Filter cat images and annotations
    cat_images = []
    cat_keypoint_annotations = []
    cat_boundingbox_annotations = []
    cat_id = None

    # Find the category ID for cats
    for category in keypoint_data['categories']:
        if category['name'].lower() == 'cat':
            cat_id = category['id']
            break

    if cat_id is None:
        raise ValueError("Cat category not found in the dataset")

    # Filter cat images and annotations
    for image in keypoint_data['images']:
        image_id = image['id']
        cat_keypoint_anns = [ann for ann in keypoint_data['annotations'] if ann['image_id'] == image_id and ann['category_id'] == cat_id]
        cat_boundingbox_anns = [ann for ann in boundingbox_data if ann['image_id'] == image_id and ann['category_id'] == cat_id]
        if cat_keypoint_anns or cat_boundingbox_anns:
            cat_images.append(image)
            cat_keypoint_annotations.extend(cat_keypoint_anns)
            cat_boundingbox_annotations.extend(cat_boundingbox_anns)

    # Create output directories
    os.makedirs(os.path.join(output_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'images'), exist_ok=True)

    # Shuffle and split the data
    random.shuffle(cat_images)
    num_images = len(cat_images)
    train_split = int(num_images * train_ratio)
    val_split = int(num_images * (train_ratio + val_ratio))

    train_images = cat_images[:train_split]
    val_images = cat_images[train_split:val_split]
    test_images = cat_images[val_split:]

    # Function to save images and annotations
    def save_split(split_name, images):
        split_keypoint_annotations = []
        split_boundingbox_annotations = []
        for image in tqdm(images, desc=f"Processing {split_name} set"):
            # Copy image
            src_path = os.path.join(keypoint_path, 'images', image['file_name'])
            dst_path = os.path.join(output_path, split_name, 'images', image['file_name'])
            shutil.copy(src_path, dst_path)

            # Collect annotations
            image_keypoint_anns = [ann for ann in cat_keypoint_annotations if ann['image_id'] == image['id']]
            image_boundingbox_anns = [ann for ann in cat_boundingbox_annotations if ann['image_id'] == image['id']]
            split_keypoint_annotations.extend(image_keypoint_anns)
            split_boundingbox_annotations.extend(image_boundingbox_anns)

        # Save annotations
        split_data = {
            'images': images,
            'keypoint_annotations': split_keypoint_annotations,
            'boundingbox_annotations': split_boundingbox_annotations,
            'categories': [cat for cat in keypoint_data['categories'] if cat['id'] == cat_id]
        }
        with open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'w') as f:
            json.dump(split_data, f)

    # Save splits
    save_split('train', train_images)
    save_split('val', val_images)
    save_split('test', test_images)

    print(f"Dataset processed and split into {len(train_images)} train, {len(val_images)} validation, and {len(test_images)} test images.")

# Usage
keypoint_path = '/workspace/Purrception/data/raw/animalpose_keypoint'
boundingbox_path = '/workspace/Purrception/data/raw/animalpose_boundingbox'
output_path = '/workspace/Purrception/data/processed'
process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path)