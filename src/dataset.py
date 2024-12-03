import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class PPGCellDataset(Dataset):
    def __init__(self, csv_file, cell_dir, start_index=None, end_index=None, transform=None, is_train=True):
        """
        Dataset for PPG cells with train/test split support.
        :param csv_file: Path to the CSV file containing labels.
        :param cell_dir: Directory containing the PPG cell files.
        :param start_index: Starting index for the dataset range.
        :param end_index: Ending index for the dataset range.
        :param transform: Optional transform to apply to the data.
        :param is_train: Boolean to indicate if the dataset is for training or testing.
        """
        self.data = pd.read_csv(csv_file)

        if start_index is not None:
            if end_index is None:
                end_index = len(self.data)
            self.data = self.data.iloc[start_index:end_index]

        # Train-test split
        split_ratio = 0.8  # 80% train, 20% test
        self.split_idx = int(len(self.data) * split_ratio)
        self.train_data = self.data[:self.split_idx]
        self.test_data = self.data[self.split_idx:]

        self.cell_dir = cell_dir
        self.transform = transform
        self.is_train = is_train

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
            print(f"Warning: PPG file {cell_path} not found. Using placeholder.")
            ppg_cells = np.zeros((10, 2, 64))  # Default 10 cells with 2x64 dimensions

        # Ensure consistent shape for each sample
        ppg_cells = torch.tensor(ppg_cells, dtype=torch.float32)

        return ppg_cells, label
