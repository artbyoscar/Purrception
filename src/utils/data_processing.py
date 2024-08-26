import json
import os
import random
import shutil
from tqdm import tqdm

def generate_bounding_box_from_keypoints(keypoints, padding=0.1):
    """Generates a bounding box from keypoints with added padding."""
    x_coords = keypoints[0::3]
    y_coords = keypoints[1::3]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width, height = x_max - x_min, y_max - y_min
    
    # Apply padding
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding
    
    return [x_min, y_min, width, height]

def find_image(image_filename, image_dirs):
    """Finds the image file in given directories."""
    for image_dir in image_dirs:
        for name in [image_filename, f"Img-{image_filename}"]:
            for ext in ['.jpg', '.jpeg', '.png']:
                full_path = os.path.join(image_dir, name.split('.')[0] + ext)
                if os.path.exists(full_path):
                    return full_path
                # Check in 'bobcat' and 'bear' subdirectories
                for subdir in ['bobcat', 'bear']:
                    sub_path = os.path.join(image_dir, subdir, name.split('.')[0] + ext)
                    if os.path.exists(sub_path):
                        return sub_path
    print(f"Warning: Image not found: {image_filename}")
    return None

def process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path, 
                                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Process and split the animal pose dataset."""
    image_dirs = [
        os.path.join(keypoint_path, 'images'),
        os.path.join(boundingbox_path, 'bndbox_image', 'bobcat'),
        os.path.join(boundingbox_path, 'bndbox_image', 'bear')
    ]
    
    # Verify directories and files
    print("Verifying directories and files:")
    for dir in image_dirs:
        print(f"- {dir} {'(exists)' if os.path.exists(dir) else '(does not exist)'}")
    
    bobcat_anno_path = os.path.join(boundingbox_path, 'bndbox_anno', 'bobcat.json')
    bear_anno_path = os.path.join(boundingbox_path, 'bndbox_anno', 'bear.json')
    for path in [bobcat_anno_path, bear_anno_path]:
        print(f"{os.path.basename(path)} {'exists' if os.path.exists(path) else 'does not exist'}")

    # Load data
    with open(os.path.join(keypoint_path, 'annotations.json'), 'r') as f:
        keypoint_data = json.load(f)
    bobcat_data = json.load(open(bobcat_anno_path)) if os.path.exists(bobcat_anno_path) else {}
    bear_data = json.load(open(bear_anno_path)) if os.path.exists(bear_anno_path) else {}

    # Process images
    cat_images = [{'id': image_id, 'file_name': filename} for image_id, filename in keypoint_data['images'].items()] # Create a list of dictionaries
    random.shuffle(cat_images)

    # Split dataset
    total_images = len(cat_images)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))

    splits = {
        'train': cat_images[:train_split],
        'val': cat_images[train_split:val_split],
        'test': cat_images[val_split:]
    }

    # Process each split
    for split, images in splits.items():
        process_split(split, images, keypoint_data, bobcat_data, bear_data, image_dirs, output_path)

def process_split(split, images, keypoint_data, bobcat_data, bear_data, image_dirs, output_path):
    """Process a single split of the dataset."""
    split_path = os.path.join(output_path, split)
    os.makedirs(os.path.join(split_path, 'images'), exist_ok=True)
    
    annotations = {'images': [], 'annotations': [], 'bobcat_annotations': [], 'bear_annotations': []}
    
    for image in tqdm(images, desc=f"Processing {split}"):
        image_filename = image['file_name']
        print(f"Looking for image: {image_filename}")
        src_image_path = find_image(image_filename, image_dirs)
        if src_image_path:
            dst_image_path = os.path.join(split_path, 'images', os.path.basename(src_image_path))
            shutil.copy(src_image_path, dst_image_path)
            annotations['images'].append(image)
            
            # Add keypoint annotations
            for ann in keypoint_data['annotations']:
                if ann['image_id'] == image['id']:
                    bbox = generate_bounding_box_from_keypoints(ann['keypoints'])
                    annotations['annotations'].append({**ann, 'bbox': bbox})
            
            # Add bobcat and bear annotations
            key_variations = [
                image_filename.lower().split('.')[0],
                f"img-{image_filename.lower().split('.')[0]}"
            ]
            for key in key_variations:
                # Convert both key and dictionary keys to lowercase for comparison
                if key in [k.lower() for k in bobcat_data.keys()]: 
                    for bbox in bobcat_data[key]:  # Access using the original key from bobcat_data
                        annotations['bobcat_annotations'].append({
                            'image_id': image['id'],
                            'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']],
                            'category_id': 'bobcat'
                        })
                if key in [k.lower() for k in bear_data.keys()]:  # Convert both key and dictionary keys to lowercase for comparison
                    for bbox in bear_data[key]:  # Access using the original key from bear_data
                        annotations['bear_annotations'].append({
                            'image_id': image['id'],
                            'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']],
                            'category_id': 'bear'
                        })

    # Save annotations for this split
    with open(os.path.join(split_path, f'{split}_annotations.json'), 'w') as f:
        json.dump(annotations, f)

    print(f"Processed {len(annotations['images'])} images for {split} set")
    print(f"Found {len(annotations['bobcat_annotations'])} bobcat annotations")
    print(f"Found {len(annotations['bear_annotations'])} bear annotations")

# Usage
keypoint_path = '/workspace/Purrception/data/raw/animalpose_keypoint'
boundingbox_path = '/workspace/Purrception/data/raw/animalpose_boundingbox'
output_path = '/workspace/Purrception/data/processed'
process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path)