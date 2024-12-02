import os
import pandas as pd
from tqdm import tqdm

def create_labels(dataset_path, output_csv):
    data = []

    # all videos in this folder are original videos, hence will be labelled 0
    original_path = os.path.join(dataset_path, "original_sequences")
    if not os.path.exists(original_path):
        print(f"Warning: {original_path} not found!")
    else:
        # os.walk() will traverse the entire directory structure, including subfolders, 
        # hence doesn't matter if there are more folders within the folder
        for root, _, files in os.walk(original_path): # a tuple (root, dirs, files) is created with os.walk
            for file in files:
                if file.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(root, file)
                    data.append({"path": video_path, "label": 0})

    # all videos in this folder are fake videos, hence will be labelled 1
    manipulated_path = os.path.join(dataset_path, "manipulated_sequences")
    if not os.path.exists(manipulated_path):
        print(f"Warning: {manipulated_path} not found!")
    else:
        for root, _, files in os.walk(manipulated_path):
            for file in files:
                if file.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(root, file)
                    data.append({"path": video_path, "label": 1})

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Labels saved to {output_csv}")
    else:
        print("No valid video files found.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    create_labels("dataset", "data/labels.csv")