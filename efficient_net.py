import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

# Build EfficientNetB7 model for deepfake detection
def build_model():
    base_model = EfficientNetB7(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)  # Binary classification: real or fake
    model = Model(inputs=base_model.input, outputs=x)
    
    return model

# Preprocess a single frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = np.expand_dims(frame_resized, axis=0)
    return preprocess_input(frame_array)

# Extract frames from video
def extract_frames(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

# Predict fake confidence for each frame
def predict_fake_confidence(model, frames):
    predictions = []
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)[0][0]
        predictions.append(prediction)
    return predictions

# Aggregate predictions to return final confidence
def aggregate_confidence(predictions):
    return np.mean(predictions)

# Main function to detect deepfake
def detect_deepfake(video_path, model):
    print("Extracting frames from video...")
    frames = extract_frames(video_path)

    if not frames:
        raise ValueError("No frames extracted from the video. Check the video path or format.")

    print("Predicting fake confidence for each frame...")
    predictions = predict_fake_confidence(model, frames)

    confidence = aggregate_confidence(predictions)
    return confidence

if __name__ == "__main__":
    video_array = [["/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0014_fake.mp4", "Deepfake"], 
        ["/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0014_real.mp4", "Real"],
        ["/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0033_fake.mp4", "Deepfake"],
        ["/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0033_real.mp4", "Real"],
        ["/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0040_fake.mp4", "Deepfake"],
        ["/home/vishwajit/Workspace/Deepfake-SIH-2024/dataset/sih_videos/0040_real.mp4", "Real"]]

    correct = 0
    for i in range(len(video_array)):
        video_path = video_array[i][0]

        model = build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # model.load_weights("path_to_trained_weights.h5")

        confidence = detect_deepfake(video_path, model)

        print(f"Deepfake Confidence Level: {confidence:.2f}")
        print("Actual: ", video_array[i][1])
        if (confidence > 0.5): 
            print("Predicted: Deepfake")
            if (video_array[i][1] == "Deepfake"): correct += 1
        else: 
            print("Predicted: Real")
            if (video_array[i][1] == "Real"): correct += 1
    print("Correct: ", correct)