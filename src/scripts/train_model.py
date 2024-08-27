import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'train', 'train_spectrograms.npy'))
    y_train = np.load(os.path.join(data_dir, 'train', 'train_labels.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_dir, 'val', 'val_spectrograms.npy'))
    y_val = np.load(os.path.join(data_dir, 'val', 'val_labels.npy'), allow_pickle=True)
    
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    
    if len(X) != len(y):
        min_samples = min(len(X), len(y))
        X = X[:min_samples]
        y = y[:min_samples]
    
    return X, y

def augment_and_balance(X, y, samples_per_class=500):
    unique_labels = np.unique(y)
    X_balanced, y_balanced = [], []
    
    for label in unique_labels:
        X_class = X[y == label]
        if len(X_class) < samples_per_class:
            multiplier = samples_per_class // len(X_class) + 1
            X_class = np.repeat(X_class, multiplier, axis=0)
            X_class = X_class[:samples_per_class]
        else:
            X_class = X_class[:samples_per_class]
        
        for i in range(len(X_class)):
            img = X_class[i]
            X_balanced.append(img)
            y_balanced.append(label)
            
            # Time shifting
            shift = np.random.randint(-20, 20)
            X_balanced.append(np.roll(img, shift, axis=1))
            y_balanced.append(label)
            
            # Frequency masking
            freq_mask = np.ones_like(img)
            mask_size = np.random.randint(5, 20)
            mask_freq = np.random.randint(0, img.shape[0] - mask_size)
            freq_mask[mask_freq:mask_freq+mask_size, :] = 0
            X_balanced.append(img * freq_mask)
            y_balanced.append(label)
            
            # Adding noise
            noise = np.random.normal(0, 0.005, img.shape)
            X_balanced.append(img + noise)
            y_balanced.append(label)
    
    return np.array(X_balanced), np.array(y_balanced)

def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_and_evaluate(X, y, unique_labels):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Augment and balance the training data
    X_train, y_train = augment_and_balance(X_train, y_train)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    model = create_model((X_train.shape[1], X_train.shape[2], 1), len(unique_labels))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
    history = model.fit(X_train, y_train_cat,
                        epochs=200,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        class_weight=class_weight_dict,
                        verbose=1)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)
    
    test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)[1]
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
    print(f'Balanced Accuracy: {balanced_acc:.4f}')
    
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    print(f'Weighted F1-score: {f1:.4f}')
    
    print(classification_report(y_true_classes, y_pred_classes, target_names=unique_labels))
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(project_root, 'models', 'confusion_matrix.png'))
    plt.close()
    
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
    plt.savefig(os.path.join(project_root, 'models', 'training_history.png'))
    plt.close()

def main():
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    print("Loading data...")
    X, y = load_data(data_dir)
    
    print("Preprocessing data...")
    unique_labels = np.unique(y)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_int[label] for label in y])
    
    print("Class distribution:")
    for i, label in enumerate(unique_labels):
        count = np.sum(y == i)
        print(f"{label}: {count} samples")
    
    print("Training and evaluating model...")
    train_and_evaluate(X, y, unique_labels)

if __name__ == "__main__":
    main()