import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'train', 'train_spectrograms.npy'))
    y_train = np.load(os.path.join(data_dir, 'train', 'train_labels.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_dir, 'val', 'val_spectrograms.npy'))
    y_val = np.load(os.path.join(data_dir, 'val', 'val_labels.npy'), allow_pickle=True)
    
    print(f"Train data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    
    print(f"Combined data shapes: X: {X.shape}, y: {y.shape}")
    print(f"X min: {X.min()}, X max: {X.max()}")
    print(f"Unique labels: {np.unique(y)}")
    
    return X, y

def check_data_consistency(X, y):
    if len(X) != len(y):
        print(f"Warning: Number of samples in X ({len(X)}) does not match number of labels in y ({len(y)})")
        # Truncate y to match X
        y = y[:len(X)]
        print(f"Truncated y to match X. New shapes: X: {X.shape}, y: {y.shape}")
    return X, y

def extract_features(X):
    features = []
    for spectrogram in X:
        # Ensure non-negative values by taking the magnitude
        S = np.abs(spectrogram)
        
        # Extract features, with error handling
        try:
            mfccs = librosa.feature.mfcc(S=S, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(S=S)
            chroma = librosa.feature.chroma_stft(S=S)
            zcr = librosa.feature.zero_crossing_rate(y=S.mean(axis=0))
            
            # Combine features
            feature = np.concatenate([
                mfccs.mean(axis=1),
                spectral_centroid.mean(axis=1),
                chroma.mean(axis=1),
                zcr.mean(axis=1)
            ])
            
            features.append(feature)
        except Exception as e:
            print(f"Error extracting features: {e}")
            print(f"Spectrogram shape: {spectrogram.shape}")
            print(f"Spectrogram min: {spectrogram.min()}, max: {spectrogram.max()}")
            # Append a zero vector if feature extraction fails
            features.append(np.zeros(13 + 1 + 12 + 1))  # MFCC + centroid + chroma + ZCR
    
    return np.array(features)

def preprocess_data(X, y):
    # Convert string labels to integers
    unique_labels = np.unique(y)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_int[label] for label in y])
    
    # Convert labels to categorical
    num_classes = len(unique_labels)
    y = to_categorical(y, num_classes=num_classes)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, unique_labels

def create_1d_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate(X, y, unique_labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_var = 1
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model = create_1d_cnn_model((X.shape[1], 1), y.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
        history = model.fit(X_train, y_train, 
                            epochs=50, 
                            batch_size=32, 
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, reduce_lr],
                            verbose=1)
        
        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold {fold_var} - Validation Accuracy: {val_accuracy:.4f}')
        
        # Generate classification report
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        print(classification_report(y_true_classes, y_pred_classes, target_names=unique_labels))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold_var}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(project_root, 'models', f'confusion_matrix_fold_{fold_var}.png'))
        plt.close()
        
        fold_var += 1

def visualize_spectrograms(X, y, num_samples=5):
    unique_labels = np.unique(y)
    fig, axes = plt.subplots(len(unique_labels), num_samples, figsize=(20, 4*len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        label_indices = np.where(y == label)[0]
        samples = np.random.choice(label_indices, min(num_samples, len(label_indices)), replace=False)
        
        for j, sample_idx in enumerate(samples):
            spectrogram = X[sample_idx]
            if len(unique_labels) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            ax.imshow(spectrogram, aspect='auto', origin='lower')
            ax.set_title(f"{label} - Sample {j+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'models', 'spectrogram_samples.png'))
    plt.close()

def main():
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_data(data_dir)
    
    # Check data consistency
    X, y = check_data_consistency(X, y)
    
    # Visualize spectrograms
    visualize_spectrograms(X, y)
    
    X = extract_features(X)
    X, y, unique_labels = preprocess_data(X, y)
    
    print("Class distribution:")
    for i, label in enumerate(unique_labels):
        count = np.sum(np.argmax(y, axis=1) == i)
        print(f"{label}: {count} samples")
    
    # Train and evaluate model
    print("Training and evaluating model...")
    train_and_evaluate(X, y, unique_labels)

if __name__ == "__main__":
    main()