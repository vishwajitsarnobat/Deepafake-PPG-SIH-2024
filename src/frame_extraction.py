import os
import cv2
import dlib
import pandas as pd
import yaml
from tqdm import tqdm

def extract_faces(video_path, output_dir, detector, frame_skip):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if faces:
                for i, face in enumerate(faces):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    roi = frame[y:y + h, x:x + w]
                    
                    if roi.size > 0:
                        face_path = os.path.join(output_dir, f"frame_{frame_count}_face_{i}.jpg")
                        cv2.imwrite(face_path, roi)
        
        frame_count += 1
    
    cap.release()

def preprocess_videos(input_csv, output_dir, frame_skip, start_index=None, end_index=None):
    detector = dlib.get_frontal_face_detector()
    data = pd.read_csv(input_csv)
    
    if start_index is not None or end_index is not None:
        data = data.iloc[start_index:end_index]
    
    for _, row in tqdm(data.iterrows(), total=len(data)):
        video_path = row["path"]
        label = str(row["label"])
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        face_output_dir = os.path.join(output_dir, label, video_name)
        extract_faces(video_path, face_output_dir, detector, frame_skip)

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    input_csv = config["dataset"]["labels_csv"]
    output_dir = config["dataset"]["frames_dir"]
    frame_skip = config.get("preprocessing", {}).get("frame_skip", 10)
    start_index = config.get("preprocessing", {}).get("start_index")
    end_index = config.get("preprocessing", {}).get("end_index")
    
    os.makedirs(output_dir, exist_ok=True)
    
    preprocess_videos(
        input_csv, 
        output_dir, 
        frame_skip, 
        start_index=start_index, 
        end_index=end_index
    )

if __name__ == "__main__":
    main()
