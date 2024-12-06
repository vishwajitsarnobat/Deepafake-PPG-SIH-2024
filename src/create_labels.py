import os
import pandas as pd

def create_labels(dataset_path, output_csv):
    data = []
    
    original_path = os.path.join(dataset_path, "original_sequences")
    if not os.path.exists(original_path):
        print(f"Warning: {original_path} not found!")
    else:
        for root, _, files in os.walk(original_path):
            for file in files:
                if file.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(root, file)
                    data.append({"path": video_path, "label": 0})
    
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
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        df.to_csv(output_csv, index=False)
        print(f"Labels saved to {output_csv}")
        print(f"Total videos: {len(df)}")
        print(f"Original videos: {len(df[df['label'] == 0])}")
        print(f"Manipulated videos: {len(df[df['label'] == 1])}")
    else:
        print("No valid video files found.")

if __name__ == "__main__":
    create_labels("dataset", "data/labels.csv")
