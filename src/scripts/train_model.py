import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
    
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    
    print(f"Combined data shapes: X: {X.shape}, y: {y.shape}")
    print(f"X min: {X.min()}, X max: {X.max()}")
    print(f"Unique labels: {np.unique(y)}")
    
    # Check for consistency
    if len(X) != len(y):
        print(f"Warning: Number of samples in X ({len(X)}) does not match number of labels in y ({len(y)})")
        # Truncate y to match X
        y = y[:len(X)]
        print(f"Truncated y to match X. New shapes: X: {X.shape}, y: {y.shape}")
    
    return X, y

def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    for i in range(len(X)):
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Time stretching
        stretched = librosa.effects.time_stretch(X[i].T, rate=0.8).T
        augmented_X.append(stretched)
        augmented_y.append(y[i])
        
        # Pitch shifting
        shifted = librosa.effects.pitch_shift(X[i].T, sr=22050, n_steps=2).T
        augmented_X.append(shifted)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1))
    return focal_loss_fixed

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 70:
        lr *= 0.5e-3
    elif epoch > 50:
        lr *= 1e-3
    elif epoch > 30:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    return lr

def train_and_evaluate(X, y, unique_labels, n_splits=5):
    num_classes = len(unique_labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, np.argmax(y, axis=1)), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Data augmentation
        X_train, y_train = augment_data(X_train, y_train)
        
        model = create_model((X_train.shape[1], X_train.shape[2]), num_classes)
        
        optimizer = Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer,
                      loss=focal_loss(gamma=2, alpha=0.25),
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        tensorboard = TensorBoard(log_dir=os.path.join(project_root, 'logs', f'fold_{fold}'))
        
        history = model.fit(X_train, y_train, 
                            epochs=100, 
                            batch_size=128, 
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, lr_scheduler, tensorboard],
                            verbose=1)
        
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold {fold} - Validation Accuracy: {val_accuracy:.4f}')
        
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        print(classification_report(y_true_classes, y_pred_classes, target_names=unique_labels, zero_division=1))
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(project_root, 'models', f'confusion_matrix_fold_{fold}.png'))
        plt.close()
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, 'models', f'training_history_fold_{fold}.png'))
        plt.close()

def main():
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    print("Loading data...")
    X, y = load_data(data_dir)
    
    print("Preprocessing data...")
    unique_labels = np.unique(y)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_int[label] for label in y])
    y = to_categorical(y)
    
    print("Class distribution:")
    for i, label in enumerate(unique_labels):
        count = np.sum(np.argmax(y, axis=1) == i)
        print(f"{label}: {count} samples")
    
    print("Training and evaluating model...")
    train_and_evaluate(X, y, unique_labels)

if __name__ == "__main__":
    main()