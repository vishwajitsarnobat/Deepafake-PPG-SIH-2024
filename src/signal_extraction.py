import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import cv2
import yaml

def extract_ppg_signals(face_images):
    # here u and v will have lesser elements than actual number of pixels
    # as the chrominance is downsampled
    signals = [] # will contain single value per frame
    for img_file in face_images:
        img = cv2.imread(img_file)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # space separates the luminance (Y) from the chrominance components (U and V)
        u, v = yuv[:, :, 1], yuv[:, :, 2] #  extracts the U and V channels (height, width, attribute)
        chrom_signal = np.mean(u) - np.mean(v) # need to change this as taking mean might not be the best
        signals.append(chrom_signal)
    return np.array(signals)

def create_ppg_cells(ppg_signals, window_size=64):

    cells = []

    for i in range(0, len(ppg_signals) - window_size, window_size):
        window = ppg_signals[i:i + window_size] # window of signals
        _, psd = welch(window, nperseg=len(window)) # Power Spectral Density

        # Ensure PSD has the same size as the window
        # 0s are added to the end of psd if needed
        if len(psd) < window_size:
            psd = np.pad(psd, (0, window_size - len(psd)), mode="constant") # padding at the end
    
        cell = np.vstack([window, psd]) # shape (2x64), vstack merges lists as rows of larger list
        cells.append(cell)

    return np.array(cells)

def process_videos_and_save(input_csv, frames_dir, ppg_cells_dir, start_index=None, end_index=None):

    data = pd.read_csv(input_csv)

    if start_index is not None or end_index is not None:
        data = data.iloc[start_index:end_index]

    os.makedirs(ppg_cells_dir, exist_ok=True)
    
    for _, row in data.iterrows():
        video_label = row["label"]
        video_name = os.path.splitext(os.path.basename(row["path"]))[0]
        
        face_images_dir = os.path.join(frames_dir, str(video_label), video_name)
        print(f"Searching for frames in: {face_images_dir}")
        
        if not os.path.exists(face_images_dir):
            print(f"ERROR: Frame directory not found: {face_images_dir}")
            continue
        
        face_images = [
            os.path.join(face_images_dir, f) 
            for f in os.listdir(face_images_dir) 
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        
        if not face_images:
            print(f"ERROR: No image frames found in {face_images_dir}")
            continue
        
        try:
            ppg_signals = extract_ppg_signals(face_images)
            ppg_cells = create_ppg_cells(ppg_signals)
            
            if len(ppg_cells) == 0:
                print(f"WARNING: No PPG cells created for {video_name}")
                continue
            
            output_file = os.path.join(ppg_cells_dir, f"{video_name}_ppg.npy")
            np.save(output_file, ppg_cells)
            print(f"PPG cells saved to {output_file}")
        
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_csv = config["dataset"]["labels_csv"]
    frames_dir = config["dataset"]["frames_dir"]
    ppg_cells_dir = config["dataset"]["ppg_cells_dir"]
    start_index = config["preprocessing"]["start_index"]
    end_index = config["preprocessing"]["end_index"]

    process_videos_and_save(input_csv, frames_dir, ppg_cells_dir, start_index=start_index, end_index=end_index)