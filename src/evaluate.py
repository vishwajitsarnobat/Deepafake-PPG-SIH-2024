import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataset import PPGCellDataset
from model import ResNetClassifier

def evaluate_model():
    """
    Evaluate the ResNet-50 model on PPG cells with comprehensive metrics.
    """
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Get evaluation parameters from config
    start_index = config["preprocessing"].get("start_index", 0)
    end_index = config["preprocessing"].get("end_index", None)
    split_ratio = config["training"].get("split_ratio", 0.8)
    
    # Custom collate function to filter out None samples
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)
    
    # Load test dataset based on the split ratio
    test_dataset = PPGCellDataset(
        csv_file=config["dataset"]["labels_csv"],
        cell_dir=config["dataset"]["ppg_cells_dir"],
        start_index=start_index,
        end_index=end_index,
        split_ratio=split_ratio,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["evaluation"]["batch_size"], 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetClassifier(num_classes=config["model"]["num_classes"])
    model.load_state_dict(torch.load(config["model"]["save_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Evaluation
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Print detailed metrics
    print(f"Total samples evaluated: {total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save confusion matrix plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'actual_labels': all_labels
    }

if __name__ == "__main__":
    evaluate_model()