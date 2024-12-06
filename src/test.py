import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import yaml

class DataLoader:
    @staticmethod
    def select_and_load_videos(labels_path, ppg_cells_folder, start_index=None, end_index=None):
        labels_df = pd.read_csv(labels_path)
        
        if start_index is not None or end_index is not None:
            labels_df = labels_df.iloc[start_index:end_index]
        
        X_all = []
        y_all = []
        
        for _, row in labels_df.iterrows():
            video_name = row['path']
            label = row.get('label', 0)
            
            filename = f"{label}_{video_name}_ppg_cells.npy"
            file_path = os.path.join(ppg_cells_folder, filename)
            
            if os.path.exists(file_path):
                ppg_cells = np.load(file_path)
                X_all.extend(ppg_cells)
                y_all.extend([label] * len(ppg_cells))
        
        return np.array(X_all), np.array(y_all)

class DeepfakeModelTester:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        y_pred = np.argmax(predictions, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        return metrics

def test_model(model_path, test_data_folder, labels_path):
    tester = DeepfakeModelTester(model_path)
    
    X_test, y_test = DataLoader.select_and_load_videos(
        labels_path, 
        test_data_folder
    )
    
    X_test = X_test.reshape(-1, 64, 64, 1) / 255.0
    
    metrics = tester.evaluate(X_test, y_test)
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["save_path"]
    test_data_folder = config["dataset"]["test_ppg_cells_dir"]
    labels_path = config["dataset"]["labels_csv"]

    test_model(model_path, test_data_folder, labels_path)
