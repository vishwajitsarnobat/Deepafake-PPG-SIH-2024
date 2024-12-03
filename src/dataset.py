import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import yaml

class PPGCellDataset(Dataset):
    def __init__(self, csv_file, cell_dir, start_index=None, end_index=None, transform=None, is_train=True):
        
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        split_ratio = config["training"].get("split_ratio", 0.8)
        self.data = pd.read_csv(csv_file)
        
        if start_index is not None:
            if end_index is None:
                end_index = len(self.data)
            
            self.data = self.data.iloc[start_index:end_index]
        
        self.split_idx = int(len(self.data) * split_ratio)
        
        self.train_data = self.data[:self.split_idx]
        self.test_data = self.data[self.split_idx:]
        
        self.cell_dir = cell_dir
        self.transform = transform
        self.is_train = is_train
        
        print(f"Total videos in range: {len(self.data)}")
        print(f"Training videos: {len(self.train_data)}")
        print(f"Testing videos: {len(self.test_data)}")

    def __len__(self):
        return len(self.train_data) if self.is_train else len(self.test_data)

    def __getitem__(self, idx):
        data = self.train_data if self.is_train else self.test_data
        
        video_path = data.iloc[idx, 0]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        label = data.iloc[idx, 1]
        
        cell_path = os.path.join(self.cell_dir, f"{video_id}_ppg.npy")
        
        try:
            ppg_cells = np.load(cell_path)
        except FileNotFoundError:
            print(f"Warning: PPG file {cell_path} not found. Skipping this sample.")
            return None
        
        ppg_cells = torch.tensor(ppg_cells, dtype=torch.float32) # 3 dim (x, 2, 64) x depends on number of pixels in frames
        
        if ppg_cells.shape[0] < 3:
            ppg_cells = ppg_cells.repeat(3, 1, 1)
        elif ppg_cells.shape[0] > 3:
            ppg_cells = ppg_cells[:3]
        
        return ppg_cells, label