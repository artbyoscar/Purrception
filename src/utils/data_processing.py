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

def find_image(image_filename, image_dirs):
    for image_dir in image_dirs:
        full_path = os.path.join(image_dir, image_filename)
        if os.path.exists(full_path):
            print(f"Found image: {full_path}")
            return full_path
        
        base_name = os.path.splitext(image_filename)[0]
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(potential_path):
                print(f"Found image with different extension: {potential_path}")
                return potential_path
    
    print(f"Image not found in any directory: {image_filename}")
    return None

def process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    image_dirs = [
        os.path.join(keypoint_path, 'images'),
        os.path.join(keypoint_path, 'images', 'images'),
        os.path.join(boundingbox_path, 'bndbox_image'),
        os.path.join(boundingbox_path, 'bndbox_image', 'bobcat')
    ]
    
    # Add subdirectories of bndbox_image
    for subdir in os.listdir(os.path.join(boundingbox_path, 'bndbox_image')):
        full_subdir_path = os.path.join(boundingbox_path, 'bndbox_image', subdir)
        if os.path.isdir(full_subdir_path):
            image_dirs.append(full_subdir_path)
    
    print("Image directories:")
    for dir in image_dirs:
        print(f"- {dir}")
        print(f"  Contents: {os.listdir(dir)[:5]}...")  # Print first 5 items
    
    # Load annotations and bobcat data
    annotations_file = os.path.join(keypoint_path, 'annotations.json')
    with open(annotations_file, 'r') as f:
        keypoint_data = json.load(f)

    bobcat_file = os.path.join(boundingbox_path, 'bndbox_anno', 'bobcat.json')
    with open(bobcat_file, 'r') as f:
        bobcat_data = json.load(f)

    print(f"Total bobcat annotations in JSON: {sum(len(anns) for anns in bobcat_data.values())}")

    # Filter cat images and annotations
    cat_images = []
    cat_keypoint_annotations = []

    for image_id, image_filename in keypoint_data['images'].items():
        cat_images.append({'id': image_id, 'file_name': image_filename})

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

    def save_split(split_name, images):
        split_keypoint_annotations = []
        split_bobcat_annotations = []
        processed_images = []
        split_missing_images = []
        for image in tqdm(images, desc=f"Processing {split_name} set"):
            image_id = image['id']
            image_filename = image['file_name']
            src_path = find_image(image_filename, image_dirs)
            if src_path:
                dst_path = os.path.join(output_path, split_name, 'images', os.path.basename(src_path))
                try:
                    shutil.copy(src_path, dst_path)
                    processed_images.append(image)
                    
                    # Collect annotations
                    image_keypoint_anns = [ann for ann in cat_keypoint_annotations if ann['image_id'] == image_id]
                    for ann in image_keypoint_anns:
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
                        print(f"Added {len(bobcat_anns)} bobcat annotations for {image_filename}")
                except Exception as e:
                    print(f"Error copying file {src_path}: {str(e)}")
                    split_missing_images.append(image_filename)
            else:
                split_missing_images.append(image_filename)

        print(f"Processed {len(processed_images)} images for {split_name} set")
        print(f"Missing {len(split_missing_images)} images for {split_name} set")

        # Save annotations
        split_data = {
            'images': processed_images,
            'keypoint_annotations': split_keypoint_annotations,
            'bobcat_annotations': split_bobcat_annotations,
            'categories': keypoint_data.get('categories', []) + [{'id': 'bobcat', 'name': 'bobcat'}]
        }
        with open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'w') as f:
            json.dump(split_data, f)

        return len(processed_images), split_missing_images

    # Save splits
    train_count, train_missing = save_split('train', train_images)
    val_count, val_missing = save_split('val', val_images)
    test_count, test_missing = save_split('test', test_images)

    total_processed = train_count + val_count + test_count
    total_missing = len(train_missing) + len(val_missing) + len(test_missing)

    print(f"Dataset processed and split into {train_count} train, {val_count} validation, and {test_count} test images.")
    print(f"Total processed images: {total_processed}")
    print(f"Total missing images: {total_missing}")
    
    # Save list of missing images
    with open(os.path.join(output_path, 'missing_images.txt'), 'w') as f:
        for missing in train_missing + val_missing + test_missing:
            f.write(f"{missing}\n")
    
    print(f"List of missing images saved to {os.path.join(output_path, 'missing_images.txt')}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total images in annotations: {len(cat_images)}")
    print(f"Total processed images: {total_processed}")
    print(f"Total missing images: {total_missing}")
    print(f"Total keypoint annotations: {len(cat_keypoint_annotations)}")
    
    # Add statistics for bobcat annotations
    total_bobcat_annotations = sum(len(split_data['bobcat_annotations']) for split_name in ['train', 'val', 'test'] 
                                   for split_data in [json.load(open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'r'))])
    print(f"Total bobcat bounding box annotations: {total_bobcat_annotations}")
    print(f"Images found percentage: {(total_processed / len(cat_images)) * 100:.2f}%")
    print(f"Bobcat annotations per processed image: {total_bobcat_annotations / total_processed:.2f}")

# Usage
keypoint_path = '/workspace/Purrception/data/raw/animalpose_keypoint'
boundingbox_path = '/workspace/Purrception/data/raw/animalpose_boundingbox'
output_path = '/workspace/Purrception/data/processed'
process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path)