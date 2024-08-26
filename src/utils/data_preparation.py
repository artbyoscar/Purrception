import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm

# Define the project root directory
PROJECT_ROOT = '/workspace/Purrception'

def audio_to_spectrogram(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Convert audio time-series to mel-spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def prepare_data(base_dir, split):
    """Prepare spectrograms and labels for a given split (train, val, or test)."""
    split_dir = os.path.join(base_dir, split)
    audio_dir = os.path.join(split_dir, 'audio')
    labels = pd.read_csv(os.path.join(split_dir, f'{split}_labels.csv'))

    spectrograms = []
    for filename in tqdm(os.listdir(audio_dir), desc=f"Processing {split} set"):
        if filename.endswith('.wav'):
            audio_path = os.path.join(audio_dir, filename)
            audio, sr = librosa.load(audio_path, sr=22050)
            spectrogram = audio_to_spectrogram(audio, sr)
            spectrograms.append(spectrogram)

    return np.array(spectrograms), labels.values.flatten()

if __name__ == "__main__":
    base_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    for split in ['train', 'val', 'test']:
        spectrograms, labels = prepare_data(base_dir, split)
        
        # Save spectrograms and labels
        np.save(os.path.join(base_dir, split, f'{split}_spectrograms.npy'), spectrograms)
        np.save(os.path.join(base_dir, split, f'{split}_labels.npy'), labels)
        
        print(f"{split} set: {len(spectrograms)} samples processed and saved.")
        