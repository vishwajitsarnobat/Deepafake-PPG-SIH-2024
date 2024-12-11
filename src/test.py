import os
import cv2
import torch
import numpy as np
import yaml
import scipy.signal
import torch.nn as nn
import matplotlib.pyplot as plt

class PPGCellExtractor:
    def __init__(self, window_size=64, low_freq=0.5, high_freq=3.0, sampling_rate=30):
        self.window_size = window_size
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
    
    def compute_vein_mask(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        vein_mask = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        kernel = np.ones((3,3), np.uint8)
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)
        return vein_mask
    
    def compute_chrom_ppg(self, roi, vein_mask=None):
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        
        if vein_mask is not None:
            combined_mask = cv2.bitwise_and(skin_mask, vein_mask)
        else:
            combined_mask = skin_mask
        
        green_mean = cv2.mean(roi[:, :, 1], mask=combined_mask)[0]
        return green_mean
    
    def band_pass_filter(self, signal_data):
        nyquist = 0.5 * self.sampling_rate
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        b, a = scipy.signal.butter(
            N=6,
            Wn=[low, high], 
            btype='band'
        )
        filtered_signal = scipy.signal.filtfilt(b, a, signal_data)
        return filtered_signal
    
    def create_ppg_cell(self, frames):
        frames = [cv2.resize(frame, (256, 256)) for frame in frames]
        ppg_signals = np.zeros((32, self.window_size))
        
        for i in range(32):
            start_x = (i % 8) * (frames[0].shape[1] // 8)
            start_y = (i // 8) * (frames[0].shape[0] // 8)
            end_x = start_x + (frames[0].shape[1] // 8)
            end_y = start_y + (frames[0].shape[0] // 8)
            
            for j, frame in enumerate(frames[:self.window_size]):
                region = frame[start_y:end_y, start_x:end_x]
                vein_mask = self.compute_vein_mask(region)
                ppg_signals[i, j] = self.compute_chrom_ppg(region, vein_mask)
        
        filtered_signals = np.zeros_like(ppg_signals)
        for i in range(32):
            filtered_signals[i] = self.band_pass_filter(ppg_signals[i])
        
        psd_signals = np.zeros_like(ppg_signals)
        for i in range(32):
            freqs, psd = scipy.signal.welch(filtered_signals[i], nperseg=self.window_size)
            if len(psd) > self.window_size:
                psd = psd[:self.window_size]
            elif len(psd) < self.window_size:
                psd = np.pad(psd, (0, self.window_size - len(psd)), mode='constant')
            psd_signals[i] = psd
        
        ppg_cell = np.vstack((filtered_signals, psd_signals))
        return ppg_cell

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
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def extract_frames(video_path, max_frames=1000):
    """
    Extract frames from a video file
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def plot_signals(ppg_signals, title="PPG Signals"):
    """
    Plot PPG signals.
    
    Parameters:
        ppg_signals (numpy.ndarray): 2D array where each row is a signal.
        title (str): Title for the plot.
    """
    num_signals = ppg_signals.shape[0]
    plt.figure(figsize=(15, 10))
    
    for i in range(num_signals):
        plt.plot(ppg_signals[i], label=f"Signal {i+1}")
    
    plt.title(title)
    plt.xlabel("Time (frames)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True)
    plt.show()

def validate_video(video_path, model_path, config_path):
    """
    Validate a video for deepfake detection
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract frames
    print(f"Extracting frames from {video_path}")
    frames = extract_frames(video_path)
    
    if not frames:
        print("No frames could be extracted. Exiting.")
        return None
    
    # Initialize PPG Cell Extractor
    extractor = PPGCellExtractor(
        window_size=config.get("preprocessing", {}).get("window_size", 64),
        sampling_rate=config.get("preprocessing", {}).get("sampling_rate", 30)
    )
    
    # Extract PPG Cells
    print("Extracting PPG Cells...")
    ppg_cells = []
    
    for i in range(0, len(frames) - extractor.window_size + 1, extractor.window_size):
        window_frames = frames[i:i+extractor.window_size]
        ppg_cell = extractor.create_ppg_cell(window_frames)
        ppg_cells.append(ppg_cell)
    
    # Convert to tensor
    ppg_cells = torch.FloatTensor(np.array(ppg_cells))

    # Visualize signals for the first window
    raw_signals = ppg_cells[0, :32].numpy()  # First half: raw signals
    filtered_signals = ppg_cells[0, 32:].numpy()  # Second half: filtered signals
    plot_signals(raw_signals, title="Raw PPG Signals")
    plot_signals(filtered_signals, title="Filtered PPG Signals")
    
    # Load model
    model = DeepfakeCNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prediction
    print("Running model prediction...")
    with torch.no_grad():
        ppg_cells = ppg_cells.to(device)
        outputs = model(ppg_cells)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Compute average predictions
        avg_pred = torch.mean(probabilities, dim=0)
        real_prob = avg_pred[0].item()
        fake_prob = avg_pred[1].item()
        
        class_pred = torch.argmax(avg_pred).item()
        
    # Print detailed results
    print(f"\n{'='*30}")
    print(f"Video Analysis Results:")
    print(f"{'='*30}")
    print(f"Video Path: {video_path}")
    print(f"Predicted Class: {'Deepfake' if class_pred == 1 else 'Real'}")
    print(f"Confidence (Real): {real_prob*100:.2f}%")
    print(f"Confidence (Fake): {fake_prob*100:.2f}%")
    
    return {
        'prediction': 'Deepfake' if class_pred == 1 else 'Real',
        'real_confidence': real_prob,
        'fake_confidence': fake_prob
    }

def main():
    # Example usageome/raj_99/Projects/SIH/plshojabc
    video_path = "/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0014_real.mp4"
    
    # Use default paths from configuration if not specified differently
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_path = config["model"]["save_path"]
    config_path = "configs/config.yaml"
    
    result = validate_video(video_path, model_path, config_path)
    
    if result:
        print("\nDetection complete.")

if __name__ == "__main__":
    main()