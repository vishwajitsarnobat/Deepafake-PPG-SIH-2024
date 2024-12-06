import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml

class PPGCellTrainer:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def load_ppg_cells(self, ppg_cells_folder):
        X = []
        video_labels = []

        for filename in os.listdir(ppg_cells_folder):
            if filename.endswith('_ppg_cells.npy'):
                label = int(filename.split('_')[0])
                ppg_cells = np.load(os.path.join(ppg_cells_folder, filename))
                X.extend(ppg_cells)
                video_labels.extend([label] * len(ppg_cells))

        return np.array(X), np.array(video_labels)

    def preprocess_data(self, X, y):
        print(f"Input X shape: {X.shape}")
        X = X.reshape(-1, 64, 64, 1) / 255.0
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test

    def create_model(self):
        inputs = tf.keras.Input(shape=(64, 64, 1))
        
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = tf.keras.layers.Flatten()(x)
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        outputs = tf.keras.layers.Dense(
            self.num_classes, 
            activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X_train, X_test, y_train, y_test):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )

        self.model = self.create_model()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping]
        )
        
        return self.model

    def save_model(self, save_path):
        if self.model:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save. Train the model first.")

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    ppg_cells_folder = config["dataset"]["ppg_cells_dir"]
    save_path = config["model"]["save_path"]
    num_classes = config["model"].get("num_classes", 2)
    
    trainer = PPGCellTrainer(num_classes=num_classes)
    
    X, y = trainer.load_ppg_cells(ppg_cells_folder)
    X_train, X_test, y_train, y_test = trainer.preprocess_data(X, y)
    
    model = trainer.train(X_train, X_test, y_train, y_test)
    
    trainer.save_model(save_path)

if __name__ == "__main__":
    main()
