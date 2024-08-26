import json
import os
import random
import shutil
from tqdm import tqdm
from pathlib import Path

def find_image(image_filename, image_dir):
    # Try exact match
    if os.path.exists(os.path.join(image_dir, image_filename)):
        return os.path.join(image_dir, image_filename)
    
    # Try matching without extension
    base_name = os.path.splitext(image_filename)[0]
    for file in os.listdir(image_dir):
        if file.startswith(base_name):
            return os.path.join(image_dir, file)
    
    return None

def process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Load the COCO format annotations for keypoints
    with open(os.path.join(keypoint_path, 'annotations.json'), 'r') as f:
        keypoint_data = json.load(f)

    print("Keypoint data structure:")
    print(json.dumps(keypoint_data, indent=2)[:500])

    # Load the COCO format annotations for bounding boxes
    boundingbox_files = [f for f in os.listdir(os.path.join(boundingbox_path, 'bndbox_anno')) if f.endswith('.json')]
    boundingbox_data = []
    for file in boundingbox_files:
        with open(os.path.join(boundingbox_path, 'bndbox_anno', file), 'r') as f:
            data = json.load(f)
            for image_filename, bboxes in data.items():
                for bbox in bboxes:
                    boundingbox_data.append({
                        'image_id': image_filename,  # Use filename as image_id
                        'category_id': file.split('.')[0],  # Use filename (without extension) as category
                        'bbox': [
                            bbox['bndbox']['xmin'],
                            bbox['bndbox']['ymin'],
                            bbox['bndbox']['xmax'] - bbox['bndbox']['xmin'],
                            bbox['bndbox']['ymax'] - bbox['bndbox']['ymin']
                        ]
                    })

    # Filter cat images and annotations
    cat_images = []
    cat_keypoint_annotations = []
    cat_boundingbox_annotations = []

    # Assuming all images in the keypoint data are cats
    for image_id, image_filename in keypoint_data['images'].items():
        cat_images.append({'id': image_id, 'file_name': image_filename})

    # Filter cat annotations
    if 'annotations' in keypoint_data:
        cat_keypoint_annotations = keypoint_data['annotations']
    
    cat_boundingbox_annotations = [ann for ann in boundingbox_data if ann['category_id'].lower() == 'cat']

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
        processed_images = []
        for image in tqdm(images, desc=f"Processing {split_name} set"):
            image_id = image['id']
            image_filename = image['file_name']
            # Find and copy image
            src_path = find_image(image_filename, os.path.join(keypoint_path, 'images'))
            if src_path:
                dst_path = os.path.join(output_path, split_name, 'images', os.path.basename(src_path))
                try:
                    shutil.copy(src_path, dst_path)
                    processed_images.append(image)
                except Exception as e:
                    print(f"Error copying file {src_path}: {str(e)}")
                    continue
            else:
                print(f"Warning: Image file not found: {image_filename}")
                continue

            # Collect annotations
            image_keypoint_anns = [ann for ann in cat_keypoint_annotations if ann['image_id'] == image_id]
            image_boundingbox_anns = [ann for ann in cat_boundingbox_annotations if ann['image_id'] == image_filename]
            split_keypoint_annotations.extend(image_keypoint_anns)
            split_boundingbox_annotations.extend(image_boundingbox_anns)

        # Save annotations
        split_data = {
            'images': processed_images,
            'keypoint_annotations': split_keypoint_annotations,
            'boundingbox_annotations': split_boundingbox_annotations,
            'categories': keypoint_data.get('categories', [])
        }
        with open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'w') as f:
            json.dump(split_data, f)

        return len(processed_images)

    # Save splits
    train_count = save_split('train', train_images)
    val_count = save_split('val', val_images)
    test_count = save_split('test', test_images)

    print(f"Dataset processed and split into {train_count} train, {val_count} validation, and {test_count} test images.")

# Usage
keypoint_path = '/workspace/Purrception/data/raw/animalpose_keypoint'
boundingbox_path = '/workspace/Purrception/data/raw/animalpose_boundingbox'
output_path = '/workspace/Purrception/data/processed'
process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path)