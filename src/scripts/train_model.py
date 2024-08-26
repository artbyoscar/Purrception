import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import librosa
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.utils.model import create_cnn_model, create_resnet_model

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'train', 'train_spectrograms.npy'))
    y_train = np.load(os.path.join(data_dir, 'train', 'train_labels.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_dir, 'val', 'val_spectrograms.npy'))
    y_val = np.load(os.path.join(data_dir, 'val', 'val_labels.npy'), allow_pickle=True)
    
    # Ensure X and y have the same number of samples for both train and val sets
    min_train_samples = min(X_train.shape[0], y_train.shape[0])
    min_val_samples = min(X_val.shape[0], y_val.shape[0])
    
    X_train = X_train[:min_train_samples]
    y_train = y_train[:min_train_samples]
    X_val = X_val[:min_val_samples]
    y_val = y_val[:min_val_samples]
    
    # Combine train and val data for re-splitting
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    
    print(f"Loaded data shapes: X: {X.shape}, y: {y.shape}")
    
    return X, y

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

def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    target_shape = X.shape[1:3]  # (128, 216)
    
    for i in range(len(X)):
        # Original sample
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Time stretching
        rate = np.random.uniform(0.8, 1.2)
        stretched = librosa.effects.time_stretch(X[i, :, :, 0], rate=rate)
        stretched = librosa.util.fix_length(stretched, size=target_shape[1], axis=1)
        stretched = np.pad(stretched, ((0, target_shape[0] - stretched.shape[0]), (0, 0)), mode='constant')
        augmented_X.append(stretched[..., np.newaxis])
        augmented_y.append(y[i])
        
        # Pitch shifting
        n_steps = np.random.randint(-2, 3)
        pitched = librosa.effects.pitch_shift(X[i, :, :, 0], sr=22050, n_steps=n_steps)
        pitched = librosa.util.fix_length(pitched, size=target_shape[1], axis=1)
        pitched = np.pad(pitched, ((0, target_shape[0] - pitched.shape[0]), (0, 0)), mode='constant')
        augmented_X.append(pitched[..., np.newaxis])
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def train_model(X_train, y_train, X_val, y_val, model_type='cnn', epochs=50):
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    if model_type == 'cnn':
        model = create_cnn_model(input_shape, num_classes)
        learning_rate = 0.001
    else:
        model = create_resnet_model(input_shape, num_classes)
        learning_rate = 0.0001

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=32, 
                        validation_data=(X_val, y_val),
                        class_weight=class_weight_dict,
                        callbacks=[early_stopping, reduce_lr])

    return model, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'], help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    args = parser.parse_args()

    data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Load data
    print("Loading data...")
    X, y = load_data(data_dir)
    print(f"Loaded data shapes: X: {X.shape}, y: {y.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(X, y)
    print(f"Preprocessed data shapes: X: {X.shape}, y: {y.shape}")
    
    # Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42)
    print(f"Train set shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation set shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Augment data
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    print(f"Augmented train set shapes: X_train_aug: {X_train_aug.shape}, y_train_aug: {y_train_aug.shape}")
    
    # Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_flat = X_train_aug.reshape(X_train_aug.shape[0], -1)
    y_train_flat = np.argmax(y_train_aug, axis=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train_flat)
    X_train_resampled = X_train_resampled.reshape(-1, *X_train_aug.shape[1:])
    y_train_resampled = to_categorical(y_train_resampled)
    print(f"Resampled train set shapes: X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")
    
    # Train final model
    print(f"Training {args.model} model for {args.epochs} epochs...")
    model, history = train_model(X_train_resampled, y_train_resampled, X_val, y_val, model_type=args.model, epochs=args.epochs)
    
    # Save model
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'cat_sound_classifier_{args.model}.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    history_path = os.path.join(model_dir, f'training_history_{args.model}.png')
    plt.savefig(history_path)
    print(f"Training history plot saved to {history_path}")

if __name__ == "__main__":
    main()