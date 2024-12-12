import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, TimeDistributed, LSTM, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class DeepfakeVideoDetector:
    def __init__(self, input_shape=(600, 600, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        # Standardize input shape
        input_shape = tuple(map(int, self.input_shape))
        
        # Base model (EfficientNetB7)
        base_model = EfficientNetB7(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        base_model.trainable = False

        # Define video input shape (frames, height, width, channels)
        video_input_shape = (30, *input_shape)
        
        # Input layer for video frames
        input_layer = Input(shape=video_input_shape)
        
        # Apply base model to each frame using TimeDistributed
        x = TimeDistributed(base_model)(input_layer)
        
        # Global Average Pooling for each frame
        x = TimeDistributed(GlobalAveragePooling2D())(x)
        
        # Temporal processing with LSTM
        x = LSTM(
            256, 
            return_sequences=False, 
            dropout=0.3, 
            recurrent_dropout=0.3
        )(x)
        
        # Normalization
        x = BatchNormalization()(x)
        
        # Fully connected layers with regularization
        x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.4)(x)

        # Output layer for binary classification
        output = Dense(self.num_classes, activation='sigmoid')(x)

        # Create and compile the model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()
            ]
        )

        return model

    def extract_frames(self, video_path, max_frames=30):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (self.input_shape[0], self.input_shape[1]))
            normalized_frame = resized_frame / 255.0
            
            frames.append(normalized_frame)
            frame_count += 1
        
        cap.release()
        
        # Pad or truncate to exactly max_frames
        if len(frames) < max_frames:
            # Pad with last frame
            frames += [frames[-1]] * (max_frames - len(frames))
        else:
            frames = frames[:max_frames]
        
        return frames

    def create_video_data_generators(self, data_dir, batch_size=32, validation_split=0.2, max_frames=30):
        manipulated_dir = os.path.join(data_dir, 'manipulated_sequences')
        original_dir = os.path.join(data_dir, 'original_sequences')
        
        manipulated_videos = []
        original_videos = []
        
        # Walk through the directories and gather video paths
        for root, _, files in os.walk(manipulated_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    manipulated_videos.append(os.path.join(root, file))
        
        for root, _, files in os.walk(original_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    original_videos.append(os.path.join(root, file))
        
        # Combine video paths and labels
        total_videos = manipulated_videos + original_videos
        total_labels = [1] * len(manipulated_videos) + [0] * len(original_videos)
        
        # Split into training and validation sets
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            total_videos, total_labels, 
            test_size=validation_split, 
            random_state=42
        )
        
        def video_generator(videos, labels, batch_size):
            while True:
                batch_videos = []
                batch_labels = []
                
                indices = np.random.choice(len(videos), batch_size, replace=False)
                
                for idx in indices:
                    try:
                        frames = self.extract_frames(videos[idx], max_frames)
                        batch_videos.append(frames)
                        batch_labels.append(labels[idx])
                    except Exception as e:
                        print(f"Error processing video {videos[idx]}: {e}")
                
                yield np.array(batch_videos), np.array(batch_labels)
        
        # Create generators
        train_gen = video_generator(train_videos, train_labels, batch_size)
        val_gen = video_generator(val_videos, val_labels, batch_size)
        
        steps_per_epoch = len(train_videos) // batch_size
        val_steps = len(val_videos) // batch_size
        
        return train_gen, val_gen, steps_per_epoch, val_steps
    
    def train(self, data_dir, epochs=50, batch_size=32, model_save_path='deepfake_detector_video_model.h5'):
        train_gen, val_gen, steps_per_epoch, val_steps = self.create_video_data_generators(
            data_dir, 
            batch_size
        )
        
        # Improved callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            min_delta=0.001
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=5, 
            min_lr=1e-6
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy', 
            save_best_only=True,
            save_weights_only=False
        )
        
        # Training
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )
        
        return history

def main():
    DATA_DIR = 'dataset'
    EPOCHS = 50
    BATCH_SIZE = 8
    MODEL_SAVE_PATH = 'deepfake_detector_video_model.keras'

    detector = DeepfakeVideoDetector()
    
    history = detector.train(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )

if __name__ == "__main__":
    main()