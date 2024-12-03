import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import PPGCellDataset
from model import ResNetClassifier

def evaluate_model():
    """
    Evaluate the ResNet-50 model on PPG cells.
    """
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    test_dataset = PPGCellDataset(
        csv_file=config["dataset"]["labels_csv"],
        cell_dir=config["dataset"]["ppg_cells_dir"],
        start_index=config["preprocessing"].get("start_index", 0),
        end_index=config["preprocessing"].get("end_index", None),
        is_train=False
    )

    # Custom collate function to skip None samples
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["evaluation"]["batch_size"], 
        shuffle=False,
        collate_fn=collate_fn
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNetClassifier(num_classes=config["model"]["num_classes"])
    model.load_state_dict(torch.load(config["model"]["save_path"], map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Flatten the 10-channel input and convert it to single-channel (1, 64, 64)
            inputs = inputs.view(inputs.size(0), -1, inputs.size(3))  # [batch_size, 64, 64]
            inputs = inputs.unsqueeze(1)  # Add channel dimension -> [batch_size, 1, 64, 64]

            # Expand single-channel input to 3 channels for ResNet
            inputs = inputs.repeat(1, 3, 1, 1)  # [batch_size, 3, 64, 64]

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model()
