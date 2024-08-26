import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import librosa
from tqdm import tqdm
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

def augment_audio(spectrogram):
    # Time stretching
    rate = np.random.uniform(0.8, 1.2)
    stretched = librosa.effects.time_stretch(spectrogram, rate=rate)
    
    # Pitch shifting
    n_steps = np.random.randint(-2, 3)
    pitched = librosa.effects.pitch_shift(stretched, sr=22050, n_steps=n_steps)
    
    return pitched

def train_model(X_train, y_train, X_val, y_val, model_type='cnn', epochs=10):
    # Get the input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    # Create the model
    if model_type == 'cnn':
        model = create_cnn_model(input_shape, num_classes)
        optimizer = Adam(learning_rate=0.001)
        batch_size = 32
    else:
        model = create_resnet_model(input_shape, num_classes)
        optimizer = Adam(learning_rate=0.0001)
        batch_size = 16

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weight_dict = dict(enumerate(class_weights))

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_val, y_val),
                        class_weight=class_weight_dict,
                        callbacks=[early_stopping],
                        verbose=0)  # Set verbose to 0 to use tqdm instead

    # Use tqdm to show progress
    for epoch in tqdm(range(epochs), desc="Training"):
        model.fit(X_train, y_train,
                  epochs=1,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val),
                  class_weight=class_weight_dict,
                  callbacks=[early_stopping],
                  verbose=0)

    return model, history

def cross_validate_model(X, y, model_type='cnn', n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X), total=n_splits, desc="Cross-validation")):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if model_type == 'cnn':
            model = create_cnn_model(X_train.shape[1:], y_train.shape[1])
            optimizer = Adam(learning_rate=0.001)
            batch_size = 32
            epochs = 10
        else:
            model = create_resnet_model(X_train.shape[1:], y_train.shape[1])
            optimizer = Adam(learning_rate=0.0001)
            batch_size = 16
            epochs = 20

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
        class_weight_dict = dict(enumerate(class_weights))

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(X_train, y_train, 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  validation_data=(X_val, y_val),
                  class_weight=class_weight_dict,
                  callbacks=[early_stopping],
                  verbose=0)

        # Evaluate the model
        score = model.evaluate(X_val, y_val, verbose=0)
        scores.append(score[1])  # Append accuracy

    return np.mean(scores), np.std(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'], help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--skip-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--skip-smote', action='store_true', help='Skip SMOTE oversampling')
    args = parser.parse_args()

    data_dir = os.path.join(project_root, 'data', 'processed')
    
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val = load_data(data_dir)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    
    print("Data loaded and preprocessed:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    if not args.skip_smote:
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
        X_train_resampled = X_train_resampled.reshape(-1, *X_train.shape[1:])
        y_train_resampled = to_categorical(y_train_resampled)
        
        print("After SMOTE:")
        print(f"X_train_resampled shape: {X_train_resampled.shape}, y_train_resampled shape: {y_train_resampled.shape}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    if not args.skip_cv:
        print("\nPerforming cross-validation...")
        mean_score, std_score = cross_validate_model(X_train_resampled, y_train_resampled, model_type=args.model, n_splits=3)
        print(f"Cross-validation result: Accuracy = {mean_score:.4f} (+/- {std_score:.4f})")
    
    print("\nTraining final model...")
    model, history = train_model(X_train_resampled, y_train_resampled, X_val, y_val, model_type=args.model, epochs=args.epochs)
    
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'cat_sound_classifier_{args.model}.h5')
    model.save(model_path)
    
    print(f"\nModel ({args.model}) trained and saved successfully to {model_path}")

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
    plt.savefig(os.path.join(model_dir, f'training_history_{args.model}.png'))
    print(f"Training history plot saved to {os.path.join(model_dir, f'training_history_{args.model}.png')}")

if __name__ == "__main__":
    main()