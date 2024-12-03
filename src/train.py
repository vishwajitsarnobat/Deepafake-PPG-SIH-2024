import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import PPGCellDataset
from model import ResNetClassifier
import numpy as np
import yaml

def train_model():

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    start_index = config["preprocessing"].get("start_index", 0)
    end_index = config["preprocessing"].get("end_index", None)
    split_ratio = config["training"].get("split_ratio", 0.8)
    
    # Create dataset with specified range and split ratio
    train_dataset = PPGCellDataset(
        csv_file=config["dataset"]["labels_csv"],
        cell_dir=config["dataset"]["ppg_cells_dir"],
        start_index=start_index,
        end_index=end_index,
        split_ratio=split_ratio,
        is_train=True
    )
    
    # Custom collate function to filter out None samples
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {total_loss/batch_count:.4f}")
    
    # Save the trained model
    os.makedirs(os.path.dirname(config["model"]["save_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["model"]["save_path"])
    print(f"Model saved to {config['model']['save_path']}")

if __name__ == "__main__":
    train_model()