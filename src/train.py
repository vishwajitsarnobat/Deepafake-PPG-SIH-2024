import os
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class PPGCellDataset(Dataset):
    def __init__(self, ppg_folder, labels_path, save_path, start_index=None, end_index=None):
        self.ppg_cells = []
        self.labels = []
        self.save_path = save_path
        labels_data = pd.read_csv(labels_path)
        if start_index is not None or end_index is not None:
            labels_data = labels_data.iloc[start_index:end_index]
        video_names = list(zip(labels_data["path"], labels_data["label"]))
        for video_name in video_names:
            base_name = os.path.splitext(os.path.basename(video_name[0]))[0]
            label = video_name[1]
            ppg_file = os.path.join(ppg_folder, f"{label}_{base_name}_ppg_cells.npy")
            if os.path.exists(ppg_file):
                ppg_cells = np.load(ppg_file)
                for ppg_cell in ppg_cells:
                    self.ppg_cells.append(ppg_cell)
                    self.labels.append(label)
        self.ppg_cells = torch.FloatTensor(np.array(self.ppg_cells))
        self.labels = torch.LongTensor(np.array(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ppg_cells[idx], self.labels[idx]

class DeepfakeCNNClassifier(nn.Module):
    def __init__(self, input_channels=64, num_classes=2):
        super(DeepfakeCNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, save_path, num_epochs, device, early_stopping=True, patience=10):
    best_val_accuracy = 0
    epochs_no_improve = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for ppg_cells, labels in train_loader:
            ppg_cells, labels = ppg_cells.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(ppg_cells)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for ppg_cells, labels in val_loader:
                ppg_cells, labels = ppg_cells.to(device), labels.to(device)
                outputs = model(ppg_cells)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        scheduler.step(val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
        if early_stopping and epochs_no_improve >= patience:
            break
    return model

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    ppg_folder = config["dataset"]["ppg_cells_dir"]
    labels_path = config["dataset"]["labels_csv"]
    start_index = config["preprocessing"]["start_index"]
    end_index = config["preprocessing"]["end_index"]
    save_path = config["model"]["save_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PPGCellDataset(ppg_folder, labels_path, save_path, start_index, end_index)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.labels.numpy(), random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, stratify=[dataset.labels[i] for i in train_indices], random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = DeepfakeCNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, save_path, num_epochs=50, device=device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_predictions = []
    test_true_labels = []
    test_correct = 0
    test_total = 0
    threshold = 0.5
    with torch.no_grad():
        for ppg_cells, labels in test_loader:
            ppg_cells, labels = ppg_cells.to(device), labels.to(device)
            outputs = model(ppg_cells)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probabilities > threshold).long()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
    test_accuracy = 100 * test_correct / test_total
    precision, recall, f1, _ = precision_recall_fscore_support(test_true_labels, test_predictions, average='binary')
    print("\nTest Results:")
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    plot_confusion_matrix(test_true_labels, test_predictions, class_names=['Real', 'Deepfake'])

if __name__ == "__main__":
    main()
