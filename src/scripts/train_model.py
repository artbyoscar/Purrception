import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.utils.model import create_resnet_model

def load_data(data_dir, subset_fraction=1.0):
    X_train = np.load(os.path.join(data_dir, 'train', 'train_spectrograms.npy'))
    y_train = np.load(os.path.join(data_dir, 'train', 'train_labels.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_dir, 'val', 'val_spectrograms.npy'))
    y_val = np.load(os.path.join(data_dir, 'val', 'val_labels.npy'), allow_pickle=True)
    
    # Combine train and val data
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    
    # Use a subset of the data
    subset_size = int(len(X) * subset_fraction)
    indices = np.random.choice(len(X), subset_size, replace=False)
    X = X[indices]
    y = y[indices]
    
    # Downsample spectrograms
    X = X[:, ::2, ::2]  # Take every other pixel in both dimensions
    
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

def create_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

class ValidationMonitor(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0], verbose=0)
        print(f"Epoch {epoch}: Validation predictions distribution: {np.bincount(np.argmax(val_pred, axis=1))}")

def train_model(X_train, y_train, X_val, y_val, model_type='cnn', epochs=200, batch_size=32, use_augmentation=True):
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    if model_type == 'cnn':
        model = create_cnn_model(input_shape, num_classes)
    else:
        model = create_resnet_model(input_shape, num_classes)

    # Use a higher initial learning rate
    initial_learning_rate = 0.01
    optimizer = Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
    X_train_res = X_train_res.reshape(-1, *X_train.shape[1:])
    y_train_res = to_categorical(y_train_res)

    # Compute class weights and adjust more aggressively
    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train_res, axis=1)), y=np.argmax(y_train_res, axis=1))
    class_weight_dict = dict(enumerate(class_weights * 2))  # Multiply weights by 2 to increase their effect

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    validation_monitor = ValidationMonitor((X_val, y_val))

    if use_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        # Fit the ImageDataGenerator on the training data
        datagen.fit(X_train_res)
        # Train the model using the data generator
        history = model.fit(datagen.flow(X_train_res, y_train_res, batch_size=batch_size),
                            steps_per_epoch=len(X_train_res) // batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, reduce_lr, validation_monitor],
                            class_weight=class_weight_dict,
                            shuffle=False)
    else:
        # Train the model without data augmentation
        history = model.fit(X_train_res, y_train_res, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_val, y_val),
                            class_weight=class_weight_dict,
                            callbacks=[early_stopping, reduce_lr, validation_monitor],
                            shuffle=False)

    return model, history

def print_class_distribution(y):
    unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id}: {count} samples")

def plot_confusion_matrix(model, X_val, y_val, model_dir, model_type):
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    confusion_matrix_path = os.path.join(model_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'], help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    args = parser.parse_args()

    data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_data(data_dir, subset_fraction=args.subset)
    X, y = preprocess_data(X, y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42)
    print(f"Train set shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation set shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    print("Training set class distribution:")
    print_class_distribution(y_train)
    print("\nValidation set class distribution:")
    print_class_distribution(y_val)
    
    # Train model
    print(f"Training {args.model} model for {args.epochs} epochs...")
    print(f"Using data augmentation: {args.augment}")
    print(f"Batch size: {args.batch_size}")
    model, history = train_model(X_train, y_train, X_val, y_val, 
                                 model_type=args.model, 
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 use_augmentation=args.augment)
    
    # Save model
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'cat_sound_classifier_{args.model}.keras')
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

    # Plot confusion matrix
    plot_confusion_matrix(model, X_val, y_val, model_dir, args.model)

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()