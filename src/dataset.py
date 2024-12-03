import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class PPGCellDataset(Dataset):
    def __init__(self, csv_file, cell_dir, start_index=None, end_index=None, split_ratio=0.8, transform=None, is_train=True):
        """
        Dataset for PPG cells with strict range filtering.
        :param csv_file: Path to the CSV file containing labels.
        :param cell_dir: Directory containing the PPG cell files.
        :param start_index: Starting index for the dataset range.
        :param end_index: Ending index for the dataset range.
        :param split_ratio: Ratio for splitting the dataset into training and testing.
        :param transform: Optional transform to apply to the data.
        :param is_train: Flag to determine whether to return training or testing data.
        """
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        
        # Apply the range filter using the start_index and end_index
        if start_index is not None:
            if end_index is None:
                end_index = len(self.data)
            
            # Strictly limit to the specified range
            self.data = self.data.iloc[start_index:end_index]
        
        # Shuffle the filtered dataset before splitting
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split dataset based on the split_ratio
        self.split_idx = int(len(self.data) * split_ratio)
        
        # Separate training and testing data
        self.train_data = self.data[:self.split_idx]
        self.test_data = self.data[self.split_idx:]
        
        self.cell_dir = cell_dir
        self.transform = transform
        self.is_train = is_train
        
        # Print dataset info for debugging
        print(f"Total videos in range: {len(self.data)}")
        print(f"Training videos: {len(self.train_data)}")
        print(f"Testing videos: {len(self.test_data)}")

    def __len__(self):
        # Return length of the train or test dataset, depending on what is being requested
        return len(self.train_data) if self.is_train else len(self.test_data)

    def __getitem__(self, idx):
        # Get training or testing data depending on the dataset being used
        data = self.train_data if self.is_train else self.test_data
        
        video_path = data.iloc[idx, 0]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        label = data.iloc[idx, 1]
        
        cell_path = os.path.join(self.cell_dir, f"{video_id}_ppg.npy")
        
        try:
            ppg_cells = np.load(cell_path)
        except FileNotFoundError:
            print(f"Warning: PPG file {cell_path} not found. Skipping this sample.")
            # Return None to allow filtering out invalid samples
            return None
        
        # Ensure 3 channels for ResNet
        ppg_cells = torch.tensor(ppg_cells, dtype=torch.float32)
        
        # Reshape to ensure consistent 3-channel input
        if ppg_cells.ndim == 2:
            ppg_cells = ppg_cells.unsqueeze(0)  # Add channel dimension if missing
        
        # Repeat to 3 channels if fewer than 3
        if ppg_cells.shape[0] < 3:
            ppg_cells = ppg_cells.repeat(3, 1, 1)
        elif ppg_cells.shape[0] > 3:
            # If more than 3 channels, take first 3
            ppg_cells = ppg_cells[:3]
        
        return ppg_cells, label