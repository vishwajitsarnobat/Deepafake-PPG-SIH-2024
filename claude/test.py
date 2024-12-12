import os
import cv2
import numpy as np
import tensorflow as tf

class DeepfakeVideoValidator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[2:5]
        self.max_frames = 30
    
    def predict_video(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (self.input_shape[0], self.input_shape[1]))
            normalized_frame = resized_frame / 255.0
            
            frames.append(normalized_frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < self.max_frames:
            raise ValueError(f"Not enough frames in video: {video_path}")
        
        frames_array = np.array(frames)
        frames_array = np.expand_dims(frames_array, axis=0)
        
        prediction = self.model.predict(frames_array)
        deepfake_prob = float(prediction[0][0])
        is_deepfake = deepfake_prob > 0.5
        
        return {
            'video_path': video_path,
            'deepfake_probability': deepfake_prob,
            'is_deepfake': is_deepfake
        }
    
    def validate_multiple_videos(self, video_dir):
        if not os.path.isdir(video_dir):
            raise ValueError(f"Not a valid directory: {video_dir}")
        
        results = []
        
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, filename)
                try:
                    result = self.predict_video(video_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return results

def main():
    MODEL_PATH = 'deepfake_detector_video_model.h5'
    validator = DeepfakeVideoValidator(MODEL_PATH)
    
    test_video_path = '/path/to/test/video.mp4'
    single_result = validator.predict_video(test_video_path)
    
    print("Video Result:")
    print(f"Video: {single_result['video_path']}")
    print(f"Deepfake Probability: {single_result['deepfake_probability']:.2%}")
    print(f"Is Deepfake: {single_result['is_deepfake']}")

if __name__ == "__main__":
    main()