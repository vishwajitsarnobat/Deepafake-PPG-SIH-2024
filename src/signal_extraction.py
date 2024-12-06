import os
import numpy as np
import cv2
import scipy.signal
import yaml
import pandas as pd

class PPGCellExtractor:
    def __init__(self, window_size=64):
        self.window_size = window_size
    
    def compute_chrom_ppg(self, roi):
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        green_mean = cv2.mean(roi[:, :, 1], mask=mask)[0]
        return green_mean
    
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
                ppg_signals[i, j] = self.compute_chrom_ppg(region)
        
        psd_signals = np.zeros_like(ppg_signals)
        for i in range(32):
            freqs, psd = scipy.signal.welch(ppg_signals[i], nperseg=self.window_size)
            
            if len(psd) > self.window_size:
                psd = psd[:self.window_size]
            elif len(psd) < self.window_size:
                psd = np.pad(psd, (0, self.window_size - len(psd)), mode='constant')
            
            psd_signals[i] = psd
        
        ppg_cell = np.vstack((ppg_signals, psd_signals))
        return ppg_cell
    
    def extract_signals_from_folder(self, frames_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
        for label_folder in os.listdir(frames_folder):
            label_path = os.path.join(frames_folder, label_folder)
            
            if not os.path.isdir(label_path):
                continue
            
            for video_folder in os.listdir(label_path):
                video_path = os.path.join(label_path, video_folder)
                
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
                    np.save(
                        os.path.join(output_folder, f"{label_folder}_{video_folder}_ppg_cells.npy"), 
                        np.array(ppg_cells)
                    )

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    frames_folder = config["dataset"]["frames_dir"]
    output_folder = config["dataset"]["ppg_cells_dir"]
    window_size = config.get("preprocessing", {}).get("window_size", 64)
    
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = PPGCellExtractor(window_size=window_size)
    extractor.extract_signals_from_folder(frames_folder, output_folder)

if __name__ == "__main__":
    main()
