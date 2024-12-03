import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import PPGCellDataset
from model import ResNetClassifier
import yaml

def train_model():
    """
    Train the ResNet-50 model on PPG cells.
    """
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    train_dataset = PPGCellDataset(
        csv_file=config["dataset"]["labels_csv"],
        cell_dir=config["dataset"]["ppg_cells_dir"],
        start_index=config["preprocessing"].get("start_index", 0),
        end_index=config["preprocessing"].get("end_index", None),
        is_train=True
    )

    # Custom collate function to skip None samples
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True,
        collate_fn=collate_fn
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing {device} for training\n")
    model = ResNetClassifier(num_classes=config["model"]["num_classes"]).to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0
        batch_count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Flatten the 10-channel input and convert it to single-channel (1, 64, 64)
            inputs = inputs.view(inputs.size(0), -1, inputs.size(3))  # [batch_size, 64, 64]
            inputs = inputs.unsqueeze(1)  # Add channel dimension -> [batch_size, 1, 64, 64]

            # Expand single-channel input to 3 channels for ResNet
            inputs = inputs.repeat(1, 3, 1, 1)  # [batch_size, 3, 64, 64]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {total_loss / batch_count:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(config["model"]["save_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["model"]["save_path"])
    print(f"Model saved to {config['model']['save_path']}")

if __name__ == "__main__":
    train_model()
