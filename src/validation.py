import os
import cv2
import torch
import numpy as np
import yaml
import scipy.signal
import torch.nn as nn
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64
import io

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
    
    def band_pass_filter(self, signal_data):
        nyquist = 0.5 * self.sampling_rate
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        b, a = scipy.signal.butter(
            N=6,
            Wn=[low, high], 
            btype='band'
        )
        filtered_signal = scipy.signal.filtfilt(b, a, signal_data)
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
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def extract_frames(video_path, max_frames=1000):
    """
    Extract frames from a video file
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def plot_signals_to_base64(ppg_signals, title="PPG Signals"):
    """
    Plot PPG signals and return as base64 encoded image
    """
    num_signals = ppg_signals.shape[0]
    plt.figure(figsize=(15, 10))
    
    for i in range(num_signals):
        plt.plot(ppg_signals[i], label=f"Signal {i+1}")
    
    plt.title(title)
    plt.xlabel("Time (frames)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode the image to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def validate_video(video_path, model_path, config_path):
    """
    Validate a video for deepfake detection
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract frames
    print(f"Extracting frames from {video_path}")
    frames = extract_frames(video_path)
    
    if not frames:
        print("No frames could be extracted. Exiting.")
        return None
    
    # Initialize PPG Cell Extractor
    extractor = PPGCellExtractor(
        window_size=config.get("preprocessing", {}).get("window_size", 64),
        sampling_rate=config.get("preprocessing", {}).get("sampling_rate", 30)
    )
    
    # Extract PPG Cells
    print("Extracting PPG Cells...")
    ppg_cells = []
    
    for i in range(0, len(frames) - extractor.window_size + 1, extractor.window_size):
        window_frames = frames[i:i+extractor.window_size]
        ppg_cell = extractor.create_ppg_cell(window_frames)
        ppg_cells.append(ppg_cell)
    
    # Convert to tensor
    ppg_cells = torch.FloatTensor(np.array(ppg_cells))

    # Prepare plots of signals
    raw_signals = ppg_cells[0, :32].numpy()  # First half: raw signals
    filtered_signals = ppg_cells[0, 32:].numpy()  # Second half: filtered signals
    raw_signals_plot = plot_signals_to_base64(raw_signals, title="Raw PPG Signals")
    filtered_signals_plot = plot_signals_to_base64(filtered_signals, title="Filtered PPG Signals")
    
    # Load model
    model = DeepfakeCNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prediction
    print("Running model prediction...")
    with torch.no_grad():
        ppg_cells = ppg_cells.to(device)
        outputs = model(ppg_cells)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Compute average predictions
        avg_pred = torch.mean(probabilities, dim=0)
        real_prob = avg_pred[0].item()
        fake_prob = avg_pred[1].item()
        
        class_pred = torch.argmax(avg_pred).item()
    
    return {
        'prediction': 'Deepfake' if class_pred == 1 else 'Real',
        'real_confidence': real_prob,
        'fake_confidence': fake_prob,
        'raw_signals_plot': raw_signals_plot,
        'filtered_signals_plot': filtered_signals_plot,
        'raw_signals': raw_signals.tolist(),
        'filtered_signals': filtered_signals.tolist()
    }

# Flask Application
app = Flask(__name__)

# Ensure a directory for uploads exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake():
    # Check if video is present in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    
    # Check if filename is empty
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Secure the filename and save the file
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)
    
    try:
        # Use default paths from configuration
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        model_path = config["model"]["save_path"]
        config_path = "configs/config.yaml"
        
        # Validate the video
        result = validate_video(video_path, model_path, config_path)
        
        # Remove the uploaded video
        os.remove(video_path)
        
        if result:
            # Return exactly: PPG signals, result, confidence levels
            return jsonify([
                result['raw_signals'].tolist() if hasattr(result['raw_signals'], 'tolist') else result['raw_signals'],
                result['prediction'],
                {
                    'real_confidence': round(result['real_confidence'], 4),
                    'fake_confidence': round(result['fake_confidence'], 4)
                }
            ])
        else:
            return jsonify({'error': 'Could not process the video'}), 500
    
    except Exception as e:
        # Remove the uploaded video in case of any error
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)