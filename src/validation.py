import os
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import scipy.signal

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

class DeepfakeCNNClassifier(nn.Module):
    def __init__(self, input_channels=64, num_classes=2):
        super(DeepfakeCNNClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def extract_frames_from_video(video_file, max_frames=300):
    """Extract frames from uploaded video file."""
    # Create a temporary directory to store frames
    temp_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    # Read the video file from memory
    video_bytes = video_file.file.read()
    video_np_array = np.frombuffer(video_bytes, np.uint8)
    
    # Open video using OpenCV
    cap = cv2.VideoCapture(video_file.filename)
    
    frames = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optional: resize frame if needed
        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)
        frame_count += 1

    cap.release()
    
    return frames

def process_video_for_ppg(frames, window_size=64):
    """Process video frames to extract PPG cells."""
    extractor = PPGCellExtractor(window_size=window_size)
    
    ppg_cells = []
    
    # Extract PPG cells with sliding window
    for i in range(0, len(frames) - window_size + 1, window_size):
        window_frames = frames[i:i+window_size]
        ppg_cell = extractor.create_ppg_cell(window_frames)
        ppg_cells.append(ppg_cell)
    
    return np.array(ppg_cells)

# Load configuration
with open("/home/vishwajit/Workspace/Deepfake-SIH-2024/configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeCNNClassifier().to(device)
model.load_state_dict(torch.load(config["model"]["save_path"], map_location=device))
model.eval()

@app.post("/validate_video/")
async def validate_video(file: UploadFile = File(...)):
    """
    Validate a video for deepfake detection.
    
    :param file: Uploaded video file
    :return: JSON response with detection result and extracted signals
    """
    # Extract frames
    frames = extract_frames_from_video(file)
    
    # Process frames to PPG cells
    ppg_cells = process_video_for_ppg(frames)
    
    # Prepare data for model
    ppg_tensor = torch.FloatTensor(ppg_cells).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(ppg_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy for JSON serialization
    probs = probabilities.cpu().numpy()
    pred = predicted.cpu().numpy()
    
    # Prepare response
    response = {
        "is_deepfake": bool(pred[0]),
        "confidence_real": float(probs[0][0]),
        "confidence_fake": float(probs[0][1]),
        "ppg_signals": {
            "raw_signals": ppg_cells[:, :32, :].tolist(),
            "filtered_signals": ppg_cells[:, 32:, :].tolist()
        }
    }
    
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)