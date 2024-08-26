import json
import os
import random
import shutil
from tqdm import tqdm

def generate_bounding_box_from_keypoints(keypoints, padding=0.1):
    """Generates a bounding box from a set of keypoints with padding."""
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
    """Finds the full path of an image given its filename and possible directories."""
    for image_dir in image_dirs:
        potential_names = [image_filename, f"Img-{image_filename}"]
        for name in potential_names:
            for ext in ['.jpg', '.jpeg', '.png']:
                full_path = os.path.join(image_dir, name.split('.')[0] + ext)
                if os.path.exists(full_path):
                    return full_path
    print(f"Warning: Image not found: {image_filename}")
    return None

def process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path, 
                                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Processes the animal pose dataset and splits it into train, validation, and test sets."""
    image_dirs = [
        os.path.join(keypoint_path, 'images'),
        os.path.join(boundingbox_path, 'bndbox_image')
    ]

    # Add all subdirectories of bndbox_image dynamically
    for root, dirs, files in os.walk(boundingbox_path):
        for dir in dirs:
            if 'bndbox_image' in dir or dir in ['bobcat', 'bear']:
                image_dirs.append(os.path.join(root, dir))
                print(f"Added image directory: {os.path.join(root, dir)}")
    
    print("Verifying image directories:")
    for dir in image_dirs:
        print(f"- {dir} {'(exists)' if os.path.exists(dir) else '(does not exist)'}")

    # Verify annotation file paths
    bobcat_anno_path = os.path.join(boundingbox_path, 'bndbox_anno', 'bobcat.json')
    bear_anno_path = os.path.join(boundingbox_path, 'bndbox_anno', 'bear.json')
    
    print("\nVerifying annotation files:")
    for path in [bobcat_anno_path, bear_anno_path]:
        print(f"{os.path.basename(path)} annotations: {'exists' if os.path.exists(path) else 'does not exist'}")

    # Load annotations
    try:
        with open(os.path.join(keypoint_path, 'annotations.json'), 'r') as f:
            keypoint_data = json.load(f)
        print("Successfully loaded keypoint annotations")
    except Exception as e:
        print(f"Error loading keypoint annotations: {str(e)}")

    try:
        with open(bobcat_anno_path, 'r') as f:
            bobcat_data = json.load(f)
        print(f"Successfully loaded bobcat annotations. Total entries: {len(bobcat_data)}")
    except Exception as e:
        print(f"Error loading bobcat annotations: {str(e)}")
        bobcat_data = {}

    try:
        with open(bear_anno_path, 'r') as f:
            bear_data = json.load(f)
        print(f"Successfully loaded bear annotations. Total entries: {len(bear_data)}")
    except Exception as e:
        print(f"Error loading bear annotations: {str(e)}")
        bear_data = {}

    # Print sample data
    print("\nSample bobcat data:")
    print(list(bobcat_data.items())[:2])
    print("\nSample bear data:")
    print(list(bear_data.items())[:2])

    # Filter cat images and annotations
    cat_images = [{'id': image_id, 'file_name': image_filename} 
                  for image_id, image_filename in keypoint_data['images'].items()]
    cat_keypoint_annotations = keypoint_data.get('annotations', [])

    # Create output directories
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split_name, 'images'), exist_ok=True)

    # Shuffle and split the data
    random.shuffle(cat_images)
    num_images = len(cat_images)
    train_split = int(num_images * train_ratio)
    val_split = int(num_images * (train_ratio + val_ratio))

    train_images = cat_images[:train_split]
    val_images = cat_images[train_split:val_split]
    test_images = cat_images[val_split:]

    def save_split(split_name, images):
        """Saves a data split to a JSON file."""
        split_keypoint_annotations = []
        split_bobcat_annotations = []
        split_bear_annotations = []
        processed_images = []
        split_missing_images = []
        for image in tqdm(images, desc=f"Processing {split_name} set"):
            image_id = image['id']
            image_filename = image['file_name']
            key_filename = image_filename.lower()

            src_path = find_image(image_filename, image_dirs)
            if src_path:
                dst_path = os.path.join(output_path, split_name, 'images', os.path.basename(src_path))
                try:
                    shutil.copy(src_path, dst_path)
                    processed_images.append(image)
                    
                    # Collect keypoint annotations
                    image_keypoint_anns = [ann for ann in cat_keypoint_annotations if ann['image_id'] == image_id]
                    for ann in image_keypoint_anns:
                        bbox = generate_bounding_box_from_keypoints(ann['keypoints'])
                        ann['bbox'] = bbox
                    split_keypoint_annotations.extend(image_keypoint_anns)

                    # Add bobcat annotations 
                    for key in [key_filename, f"img-{key_filename}", key_filename.split('.')[0]]:
                        if key in bobcat_data:
                            bobcat_anns = bobcat_data[key]
                            for bbox in bobcat_anns:
                                split_bobcat_annotations.append({
                                    'image_id': image_id,
                                    'bbox': [
                                        bbox['xmin'], 
                                        bbox['ymin'],
                                        bbox['xmax'] - bbox['xmin'],
                                        bbox['ymax'] - bbox['ymin']
                                    ],
                                    'category_id': 'bobcat'
                                })
                            print(f"Added {len(bobcat_anns)} bobcat annotations for {key}")
                            break

                    # Add bear annotations
                    for key in [key_filename, f"img-{key_filename}", key_filename.split('.')[0]]:
                        if key in bear_data:
                            bear_anns = bear_data[key]
                            for bbox in bear_anns:
                                split_bear_annotations.append({
                                    'image_id': image_id,
                                    'bbox': [
                                        bbox['xmin'], 
                                        bbox['ymin'],
                                        bbox['xmax'] - bbox['xmin'],
                                        bbox['ymax'] - bbox['ymin']
                                    ],
                                    'category_id': 'bear'
                                })
                            print(f"Added {len(bear_anns)} bear annotations for {key}")
                            break

                except Exception as e:
                    print(f"Error copying file {src_path}: {str(e)}")
                    split_missing_images.append(image_filename)
            else:
                split_missing_images.append(image_filename)

        print(f"Processed {len(processed_images)} images for {split_name} set")
        print(f"Missing {len(split_missing_images)} images for {split_name} set")
        print(f"Total bobcat annotations for {split_name}: {len(split_bobcat_annotations)}")
        print(f"Total bear annotations for {split_name}: {len(split_bear_annotations)}")

        # Save annotations
        split_data = {
            'images': processed_images,
            'keypoint_annotations': split_keypoint_annotations,
            'bobcat_annotations': split_bobcat_annotations,
            'bear_annotations': split_bear_annotations,
            'categories': keypoint_data.get('categories', []) + [{'id': 'bobcat', 'name': 'bobcat'}, {'id': 'bear', 'name': 'bear'}]
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
    
    # Calculate total annotations from saved split files
    total_bobcat_annotations = 0
    total_bear_annotations = 0
    for split_name in ['train', 'val', 'test']:
        with open(os.path.join(output_path, split_name, f'{split_name}_annotations.json'), 'r') as f:
            split_data = json.load(f)
            total_bobcat_annotations += len(split_data['bobcat_annotations'])
            total_bear_annotations += len(split_data['bear_annotations'])

    print(f"Total bobcat bounding box annotations: {total_bobcat_annotations}")
    print(f"Total bear bounding box annotations: {total_bear_annotations}")
    print(f"Images found percentage: {(total_processed / len(cat_images)) * 100:.2f}%")
    if total_processed > 0:
        print(f"Bobcat annotations per processed image: {total_bobcat_annotations / total_processed:.2f}")
        print(f"Bear annotations per processed image: {total_bear_annotations / total_processed:.2f}")
    else:
        print("No images were processed.")

# Usage
keypoint_path = '/workspace/Purrception/data/raw/animalpose_keypoint'
boundingbox_path = '/workspace/Purrception/data/raw/animalpose_boundingbox'
output_path = '/workspace/Purrception/data/processed'
process_animal_pose_dataset(keypoint_path, boundingbox_path, output_path)