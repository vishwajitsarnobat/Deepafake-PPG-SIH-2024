import os
import numpy as np
import cv2
import scipy.signal
import yaml
import pandas as pd

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
    
    def band_pass_filter(self, signal):
        nyquist = 0.5 * self.sampling_rate
        
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        b, a = scipy.signal.butter(
            N=6,
            Wn=[low, high], 
            btype='band'
        )
        
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        
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
    
    def extract_signals_from_folder(self, frames_folder, output_folder, labels_path, start_index=None, end_index=None):
        os.makedirs(output_folder, exist_ok=True)
        
        video_data = pd.read_csv(labels_path)
        video_names = list(zip(video_data["path"], video_data["label"]))
        
        if start_index is not None or end_index is not None:
            video_names = video_names[start_index:end_index]
                
        for video_name in video_names:
            base_name = os.path.splitext(os.path.basename(video_name[0]))[0]
            label = video_name[1]

            video_path = os.path.join(frames_folder, str(label), base_name)
            
            if os.path.exists(video_path):
                output_file = os.path.join(output_folder, f"{label}_{base_name}_ppg_cells.npy")
                
                if os.path.exists(output_file):
                    print(f"Skipping {video_path}, PPG cells already exist: {output_file}")
                    continue
                
                frame_files = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ])
                
                if not frame_files:
                    print(f"No frames found in {video_path}")
                    continue
                
                frames = [cv2.imread(f) for f in frame_files]
                ppg_cells = []
                
                for i in range(0, len(frames) - self.window_size + 1, self.window_size):
                    window_frames = frames[i:i+self.window_size]
                    ppg_cell = self.create_ppg_cell(window_frames)
                    ppg_cells.append(ppg_cell)
                
                if ppg_cells:
                    np.save(output_file, np.array(ppg_cells))

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    frames_folder = config["dataset"]["frames_dir"]
    output_folder = config["dataset"]["ppg_cells_dir"]
    labels_path = config["dataset"]["labels_csv"]
    window_size = config.get("preprocessing", {}).get("window_size", 64)
    sampling_rate = config.get("preprocessing", {}).get("sampling_rate", 30)
    start_index = config["preprocessing"]["start_index"]
    end_index = config["preprocessing"]["end_index"]
    
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = PPGCellExtractor(
        window_size=window_size, 
        sampling_rate=sampling_rate
    )
    extractor.extract_signals_from_folder(
        frames_folder, 
        output_folder, 
        labels_path,
        start_index,
        end_index
    )

if __name__ == "__main__":
    main()