import os
import numpy as np
import tensorflow as tf
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class PPGCellTester:
    def __init__(self, model_path, num_classes=2):
        self.model = tf.keras.models.load_model(model_path)
        self.num_classes = num_classes

    def load_test_data(self, ppg_cells_folder):
        X = []
        video_labels = []

        for filename in os.listdir(ppg_cells_folder):
            if filename.endswith('_ppg_cells.npy'):
                label = int(filename.split('_')[0])
                ppg_cells = np.load(os.path.join(ppg_cells_folder, filename))
                
                print(f"Loading {filename}, shape: {ppg_cells.shape}")
                
                X.extend(ppg_cells)
                video_labels.extend([label] * len(ppg_cells))

        X = np.array(X)
        print(f"Total X shape before reshape: {X.shape}")
        
        X = X.reshape(-1, 64, 64, 1) / 255.0
        y = np.array(video_labels)

        return X, y

    def aggregate_predictions(self, cell_predictions):
        if len(cell_predictions.shape) > 1 and cell_predictions.shape[1] > 1:
            return cell_predictions
        
        probabilities = cell_predictions if cell_predictions.max() <= 1 else tf.nn.softmax(cell_predictions)
        
        video_prediction = np.mean(probabilities, axis=0)
        
        return video_prediction.reshape(1, -1)

    def predict(self, X):
        print(f"Input X shape: {X.shape}")
        
        cell_predictions = self.model.predict(X)
        
        print(f"Cell predictions shape: {cell_predictions.shape}")
        
        video_prediction = self.aggregate_predictions(cell_predictions)
        
        print(f"Video prediction shape: {video_prediction.shape}")
        
        return np.argmax(video_prediction, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        return metrics

    def generate_detailed_report(self, metrics):
        report = "Deep Fake Detection Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Overall Accuracy: {metrics['accuracy']:.2%}\n"
        report += f"Precision: {metrics['precision']:.2%}\n"
        report += f"Recall: {metrics['recall']:.2%}\n"
        report += f"F1 Score: {metrics['f1_score']:.2%}\n\n"
        
        report += "Classification Report:\n"
        report += metrics['classification_report'] + "\n\n"
        
        report += "Confusion Matrix:\n"
        report += str(metrics['confusion_matrix']) + "\n"
        
        return report

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_path = config["model"]["save_path"]
    ppg_cells_folder = config["dataset"]["ppg_cells_dir"]
    report_path = config["results"]["report_path"]
    num_classes = config["model"].get("num_classes", 2)
    
    tester = PPGCellTester(model_path, num_classes=num_classes)
    
    X_test, y_test = tester.load_test_data(ppg_cells_folder)
    
    metrics = tester.evaluate(X_test, y_test)
    
    report = tester.generate_detailed_report(metrics)
    print(report)
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
