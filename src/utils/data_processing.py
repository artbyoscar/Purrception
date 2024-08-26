import json
import os
import random
import shutil
from tqdm import tqdm

def generate_bounding_box_from_keypoints(keypoints, padding=0.1):
    x_coords = keypoints[0::3]
    y_coords = keypoints[1::3]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Add padding
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def find_image(image_filename, image_dir):
    # Try exact match
    if os.path.exists(os.path.join(image_dir, image_filename)):
        return os.path.join(image_dir, image_filename)
    
    # Try matching without extension
    base_name = os.path.splitext(image_filename)[0]
    for ext in ['.jpg', '.jpeg']:
        if os.path.exists(os.path.join(image_dir, base_name + ext)):
            return os.path.join(image_dir, base_name + ext)
    
    return None

def process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    image_path = os.path.join(keypoint_path, 'images', 'images')
    
    print("Contents of image directory:")
    print(os.listdir(image_path)[:10])  # Print first 10 files for brevity
    
    # Load the COCO format annotations for keypoints
    annotations_file = os.path.join(keypoint_path, 'annotations.json')
    with open(annotations_file, 'r') as f:
        keypoint_data = json.load(f)

    print("Keypoint data structure:")
    print(json.dumps(keypoint_data, indent=2)[:500])

    # Load bobcat bounding box data
    bobcat_file = os.path.join(boundingbox_path, 'bndbox_anno', 'bobcat.json')
    with open(bobcat_file, 'r') as f:
        bobcat_data = json.load(f)

    # Filter cat images and annotations
    cat_images = []
    cat_keypoint_annotations = []

    # Assuming all images in the keypoint data are cats
    for image_id, image_filename in keypoint_data['images'].items():
        cat_images.append({'id': image_id, 'file_name': image_filename})

    # Filter cat annotations
    if 'annotations' in keypoint_data:
        cat_keypoint_annotations = keypoint_data['annotations']

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

    missing_images = []

    # Function to save images and annotations
    def save_split(split_name, images):
        split_keypoint_annotations = []
        split_bobcat_annotations = []
        processed_images = []
        for image in tqdm(images, desc=f"Processing {split_name} set"):
            image_id = image['id']
            image_filename = image['file_name']
            # Find and copy image
            src_path = find_image(image_filename, image_path)
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
                missing_images.append(image_filename)
                continue

            # Collect annotations
            image_keypoint_anns = [ann for ann in cat_keypoint_annotations if ann['image_id'] == image_id]
            for ann in image_keypoint_anns:
                # Generate bounding box from keypoints
                bbox = generate_bounding_box_from_keypoints(ann['keypoints'])
                ann['bbox'] = bbox
            split_keypoint_annotations.extend(image_keypoint_anns)

            # Add bobcat annotations if available
            if image_filename in bobcat_data:
                bobcat_anns = bobcat_data[image_filename]
                for bbox in bobcat_anns:
                    split_bobcat_annotations.append({
                        'image_id': image_id,
                        'bbox': [
                            bbox['bndbox']['xmin'],
                            bbox['bndbox']['ymin'],
                            bbox['bndbox']['xmax'] - bbox['bndbox']['xmin'],
                            bbox['bndbox']['ymax'] - bbox['bndbox']['ymin']
                        ],
                        'category_id': 'bobcat'
                    })

        # Save annotations
        split_data = {
            'images': processed_images,
            'keypoint_annotations': split_keypoint_annotations,
            'bobcat_annotations': split_bobcat_annotations,
            'categories': keypoint_data.get('categories', []) + [{'id': 'bobcat', 'name': 'bobcat'}]
        }
        with open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'w') as f:
            json.dump(split_data, f)

        return len(processed_images)

    # Save splits
    train_count = save_split('train', train_images)
    val_count = save_split('val', val_images)
    test_count = save_split('test', test_images)

    print(f"Dataset processed and split into {train_count} train, {val_count} validation, and {test_count} test images.")
    
    # Save list of missing images
    with open(os.path.join(output_path, 'missing_images.txt'), 'w') as f:
        for missing in missing_images:
            f.write(f"{missing}\n")
    
    print(f"Total missing images: {len(missing_images)}")
    print(f"List of missing images saved to {os.path.join(output_path, 'missing_images.txt')}")

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total images in annotations: {len(cat_images)}")
    print(f"Total processed images: {train_count + val_count + test_count}")
    print(f"Total keypoint annotations: {len(cat_keypoint_annotations)}")
    
    # Add statistics for bobcat annotations
    total_bobcat_annotations = 0
    for split_name in ['train', 'val', 'test']:
        with open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'r') as f:
            split_data = json.load(f)
            total_bobcat_annotations += len(split_data.get('bobcat_annotations', []))

    print(f"Total bobcat bounding box annotations: {total_bobcat_annotations}")

# Usage
keypoint_path = '/workspace/Purrception/data/raw/animalpose_keypoint'
boundingbox_path = '/workspace/Purrception/data/raw/animalpose_boundingbox'
output_path = '/workspace/Purrception/data/processed'
process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path)