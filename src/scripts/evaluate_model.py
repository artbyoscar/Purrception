import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.scripts.train_model import preprocess_data

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_and_preprocess_data(data_dir):
    X_test = np.load(os.path.join(data_dir, 'test', 'test_spectrograms.npy'))
    y_test = np.load(os.path.join(data_dir, 'test', 'test_labels.npy'), allow_pickle=True)
    
    print("Before preprocessing:")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"y_test dtype: {y_test.dtype}")
    print(f"Unique labels in y_test: {np.unique(y_test)}")
    
    # Ensure X and y have the same number of samples
    min_samples = min(X_test.shape[0], y_test.shape[0])
    X_test = X_test[:min_samples]
    y_test = y_test[:min_samples]
    
    X_test, y_test = preprocess_data(X_test, y_test)
    
    print("\nAfter preprocessing:")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Unique values in y_test: {np.unique(y_test)}")
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    # Calculate and print ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
    print(f"\nWeighted ROC AUC Score: {roc_auc:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nConfusion matrix saved to {confusion_matrix_path}")

    # Calculate and print overall accuracy
    accuracy = np.sum(y_pred_classes == y_true_classes) / len(y_true_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    return y_pred, y_true_classes

def print_model_summary(model):
    print("\nModel Summary:")
    model.summary()

    print("\nModel Layers:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'output_shape'):
            output_shape = layer.output_shape
        elif hasattr(layer, 'output'):
            output_shape = layer.output.shape
        else:
            output_shape = "Unknown"
        print(f"Layer {i}: {layer.__class__.__name__}, Output Shape: {output_shape}")

def plot_sample_predictions(X_test, y_true, y_pred, num_samples=5):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 3*num_samples))
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(X_test[idx, :, :, 0], aspect='auto', cmap='viridis')
        plt.title(f"True: {y_true[idx]}, Predicted: {np.argmax(y_pred[idx])}")
        plt.colorbar()
    
    results_dir = os.path.join(project_root, 'results')
    sample_predictions_path = os.path.join(results_dir, 'sample_predictions.png')
    plt.tight_layout()
    plt.savefig(sample_predictions_path)
    plt.close()
    print(f"\nSample predictions plot saved to {sample_predictions_path}")

def main():
    # Load and preprocess test data
    data_dir = os.path.join(project_root, 'data', 'processed')
    X_test, y_test = load_and_preprocess_data(data_dir)

    # Load the trained model
    model_path = os.path.join(project_root, 'models', 'cat_sound_classifier_cnn.h5')
    model = load_model(model_path)

    # Print model summary
    print_model_summary(model)

    # Evaluate the model
    y_pred, y_true = evaluate_model(model, X_test, y_test)

    # Plot sample predictions
    plot_sample_predictions(X_test, y_true, y_pred)

    print("Evaluation completed. Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()