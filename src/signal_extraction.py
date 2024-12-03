import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import cv2
import yaml

def extract_ppg_signals(face_images):
    """
    Extract PPG signals from face images.
    :param face_images: List of face image paths.
    :return: Array of raw PPG signals.
    """
    signals = []
    for img_file in face_images:
        img = cv2.imread(img_file)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        u, v = yuv[:, :, 1], yuv[:, :, 2]
        chrom_signal = np.mean(u) - np.mean(v)
        signals.append(chrom_signal)
    return np.array(signals)

def create_ppg_cells(ppg_signals, window_size=64):
    """
    Create PPG cells by combining raw signals and their PSD.
    :param ppg_signals: Raw PPG signals.
    :param window_size: Number of frames per window.
    :return: List of PPG cells.
    """
    cells = []

    for i in range(0, len(ppg_signals) - window_size, window_size):
        window = ppg_signals[i:i + window_size]
        _, psd = welch(window, nperseg=len(window))

        # Ensure PSD has the same size as the window
        if len(psd) < window_size:
            psd = np.pad(psd, (0, window_size - len(psd)), mode="constant")

        cell = np.vstack([window, psd])
        cells.append(cell)

    return np.array(cells)

def process_videos_and_save(input_csv, frames_dir, ppg_cells_dir, start_index=None, end_index=None):
    """
    Process videos to extract PPG cells and save individual .npy files for each video.
    :param input_csv: Path to the labels CSV file.
    :param frames_dir: Path to the directory containing extracted frames.
    :param ppg_cells_dir: Path to save individual PPG cell files.
    :param start_index: Starting index for selecting videos.
    :param end_index: Ending index for selecting videos.
    """
    data = pd.read_csv(input_csv)

    # Filter the range of videos
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
            if f.endswith((".jpg", ".png", ".jpeg"))  # Support more image types
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
            
            # Save individual PPG cell file for each video
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