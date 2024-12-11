import os
import numpy as np
import cv2
import torch
import yaml
import tempfile
import scipy.signal

class VideoProcessor:
    def __init__(self, config_path="configs/config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Load model parameters
        self.window_size = self.config.get("preprocessing", {}).get("window_size", 64)
        self.sampling_rate = self.config.get("preprocessing", {}).get("sampling_rate", 30)
        
        # Frequency band parameters
        self.low_freq = 0.5
        self.high_freq = 3.0
        
        # Load trained model
        from model import DeepfakeCNNClassifier
        self.model = DeepfakeCNNClassifier()
        self.model.load_state_dict(torch.load('best_deepfake_model.pth', map_location=torch.device('cpu')))
        self.model.eval()
    
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
            N=4,
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
    
    def process_video(self, video_bytes):
        # Save video bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
        
        try:
            # Open video and extract frames
            cap = cv2.VideoCapture(temp_video_path)
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            # Remove temporary video file
            os.unlink(temp_video_path)
            
            # If not enough frames, return None
            if len(frames) < self.window_size:
                return None
            
            # Prepare PPG cells
            ppg_cells = []
            for i in range(0, len(frames) - self.window_size + 1, self.window_size):
                window_frames = frames[i:i+self.window_size]
                ppg_cell = self.create_ppg_cell(window_frames)
                ppg_cells.append(ppg_cell)
            
            # Convert to tensor and process
            ppg_cells_tensor = torch.FloatTensor(np.array(ppg_cells))
            
            # Predict
            with torch.no_grad():
                outputs = self.model(ppg_cells_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Prepare results
            results = {
                'prediction': 'Fake' if predicted[0] == 1 else 'Real',
                'confidence': probabilities[0][predicted[0]].item(),
                'ppg_signals': {
                    'raw_signals': ppg_cells[0][0].tolist(),
                    'filtered_signals': ppg_cells[0][0].tolist(),
                    'psd_signals': ppg_cells[0][1].tolist()
                }
            }
            
            return results
        
        except Exception as e:
            print(f"Error processing video: {e}")
            return None