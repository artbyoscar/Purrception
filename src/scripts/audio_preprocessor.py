import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    
    return np.concatenate([mfccs, spectral_centroid, chroma])

def process_dataset(dataset_dir):
    features = []
    labels = []
    
    for root, _, files in os.walk(dataset_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(dataset_dir)}"):
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    feature = preprocess_audio(file_path)
                    features.append(feature.flatten())
                    labels.append(os.path.basename(root))  # Assuming folder name is the label
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(features), pd.Series(labels)

# Process all datasets
all_features = []
all_labels = []

for _, folder_name in datasets:
    dataset_dir = os.path.join('data', folder_name)
    features, labels = process_dataset(dataset_dir)
    all_features.append(features)
    all_labels.append(labels)

# Combine all datasets
combined_features = pd.concat(all_features, ignore_index=True)
combined_labels = pd.concat(all_labels, ignore_index=True)

# Save preprocessed data
combined_features.to_csv('data/preprocessed_features.csv', index=False)
combined_labels.to_csv('data/preprocessed_labels.csv', index=False)

print("Preprocessing completed. Data saved to 'data/preprocessed_features.csv' and 'data/preprocessed_labels.csv'")