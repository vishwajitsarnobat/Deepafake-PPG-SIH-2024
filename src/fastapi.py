from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import sys
import os

# Ensure the directory containing the validation script is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validation import VideoProcessor

class DeepfakeResponse(BaseModel):
    prediction: str
    confidence: float
    ppg_signals: dict

app = FastAPI(
    title="Deepfake Video Detection API", 
    description="API for detecting deepfake videos using PPG signal analysis"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the video processor
video_processor = VideoProcessor()

@app.post("/detect", response_model=DeepfakeResponse)
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Endpoint to detect deepfake in an uploaded video file.
    
    - Accepts a video file upload
    - Processes the video using PPG signal analysis
    - Returns prediction, confidence, and extracted signals
    """
    # Check if file is uploaded
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read file contents
    contents = await file.read()
    
    # Process video
    result = video_processor.process_video(contents)
    
    # Check if processing was successful
    if result is None:
        raise HTTPException(status_code=400, detail="Unable to process video")
    
    return result

def main():
    """
    Run the FastAPI application
    """
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )

if __name__ == "__main__":
    main()