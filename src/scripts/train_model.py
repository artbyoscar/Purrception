# File: src/scripts/train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.utils.model import create_cnn_model

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'train', 'train_spectrograms.npy'))
    y_train = np.load(os.path.join(data_dir, 'train', 'train_labels.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_dir, 'val', 'val_spectrograms.npy'))
    y_val = np.load(os.path.join(data_dir, 'val', 'val_labels.npy'), allow_pickle=True)
    
    # Ensure X and y have the same number of samples
    min_train_samples = min(X_train.shape[0], y_train.shape[0])
    min_val_samples = min(X_val.shape[0], y_val.shape[0])
    
    X_train = X_train[:min_train_samples]
    y_train = y_train[:min_train_samples]
    X_val = X_val[:min_val_samples]
    y_val = y_val[:min_val_samples]
    
    return X_train, y_train, X_val, y_val

def preprocess_data(X, y):
    # Add channel dimension to X
    X = X[..., np.newaxis]
    
    # Normalize the input
    X = (X - X.min()) / (X.max() - X.min())
    
    # Convert string labels to integers if necessary
    if y.dtype == object:
        unique_labels = np.unique(y)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_to_int[label] for label in y])
    
    # Convert labels to categorical
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes=num_classes)
    
    return X, y

def train_model(X_train, y_train, X_val, y_val):
    # Get the input shape and number of classes
    input_shape = X_train.shape[1:]  # Now includes the channel dimension
    num_classes = y_train.shape[1]

    # Create the model
    model = create_cnn_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=32, 
                        validation_data=(X_val, y_val))

    return model, history

def main():
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Load data
    X_train, y_train, X_val, y_val = load_data(data_dir)
    
    print("Data loaded:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"y_train dtype: {y_train.dtype}, y_val dtype: {y_val.dtype}")
    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Unique labels in y_val: {np.unique(y_val)}")
    
    # Preprocess data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    
    print("\nAfter preprocessing:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"Sample X_train range: {X_train.min()} to {X_train.max()}")
    print(f"Sample y_train: {y_train[0]}")
    print(f"Unique values in y_train: {np.unique(y_train)}")
    print(f"Unique values in y_val: {np.unique(y_val)}")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Save model
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'cat_sound_classifier.h5'))
    
    print("Model trained and saved successfully.")
    
if __name__ == "__main__":
    main()