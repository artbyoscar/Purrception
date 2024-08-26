import os
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define the datasets
datasets = [
    ('1nxOWNJwSEMhhh0KrPy2o9LuICza2miKr', 'catmeows_dataset1'),
    ('1M-PH4szGZd7N9qN26dz1CgqUqGcZ08vG', 'catmeows_dataset2'),
    ('1tYuzkFVJ9FYUE3UmOt7MVdQ_DLCuHejp', 'catmeows_dataset3')
]

def preprocess_audio(file_path, target_sr=22050, duration=5):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # Pad or trim to target duration
    target_length = int(duration * target_sr)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=target_sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr)
    
    return np.concatenate([mfccs, spectral_centroid, chroma]), audio

def process_dataset(dataset_dir):
    features = []
    labels = []
    audio_data = []
    file_paths = []
    
    for root, _, files in os.walk(dataset_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(dataset_dir)}"):
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    feature, audio = preprocess_audio(file_path)
                    features.append(feature.flatten())
                    labels.append(os.path.basename(root))  # Assuming folder name is the label
                    audio_data.append(audio)
                    file_paths.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(features), pd.Series(labels), audio_data, file_paths

def main():
    # Process all datasets
    all_features = []
    all_labels = []
    all_audio = []
    all_file_paths = []

    print("Current working directory:", os.getcwd())
    data_dir = '/workspace/Purrception/data/raw'
    print(f"Contents of {data_dir}:")
    try:
        print(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"Directory {data_dir} not found.")

    for _, folder_name in datasets:
        dataset_dir = os.path.join(data_dir, folder_name)
        print(f"Checking directory: {dataset_dir}")
        if os.path.exists(dataset_dir):
            print(f"Processing {dataset_dir}")
            features, labels, audio, file_paths = process_dataset(dataset_dir)
            all_features.append(features)
            all_labels.append(labels)
            all_audio.extend(audio)
            all_file_paths.extend(file_paths)
        else:
            print(f"Warning: Directory {dataset_dir} not found. Skipping.")

    if not all_features:
        print("No datasets were processed. Please ensure the data has been downloaded.")
        return

    # Combine all datasets
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.Series(pd.concat(all_labels, ignore_index=True))

    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test, audio_train, audio_test, paths_train, paths_test = train_test_split(
        combined_features, combined_labels, all_audio, all_file_paths, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val, audio_train, audio_val, paths_train, paths_val = train_test_split(
        X_train, y_train, audio_train, paths_train, test_size=0.25, random_state=42
    )

    # Save preprocessed data
    base_output_dir = '/workspace/Purrception/data/processed'
    for split, X, y, audio, paths in [
        ('train', X_train, y_train, audio_train, paths_train),
        ('val', X_val, y_val, audio_val, paths_val),
        ('test', X_test, y_test, audio_test, paths_test)
    ]:
        output_dir = os.path.join(base_output_dir, split)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'audio'), exist_ok=True)
        
        # Save features and labels
        X.to_csv(os.path.join(output_dir, f'{split}_features.csv'), index=False)
        y.to_csv(os.path.join(output_dir, f'{split}_labels.csv'), index=False)
        
        # Save audio files
        for aud, path in zip(audio, paths):
            filename = os.path.basename(path)
            sf.write(os.path.join(output_dir, 'audio', filename), aud, 22050)

    print(f"Preprocessing completed. Data saved to {base_output_dir}")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

if __name__ == "__main__":
    main()