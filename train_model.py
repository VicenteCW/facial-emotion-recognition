"""
Train emotion recognition model
Supports FER2013 dataset or custom dataset
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

from emotion_recognition import EmotionRecognizer


class ModelTrainer:
    """Train emotion recognition model"""
    
    def __init__(self, data_path=None):
        """
        Initialize trainer
        
        Args:
            data_path: Path to dataset
        """
        self.recognizer = EmotionRecognizer()
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_fer2013_data(self, csv_path):
        """
        Load FER2013 dataset from CSV
        Download from: https://www.kaggle.com/datasets/msambare/fer2013
        
        Args:
            csv_path: Path to fer2013.csv
        """
        print("Loading FER2013 dataset...")
        
        data_X = []
        data_y = []
        
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # Skip header
                    continue
                
                parts = line.strip().split(',')
                emotion = int(parts[0])
                pixels = np.array([int(p) for p in parts[1].split()], dtype=np.uint8)
                image = pixels.reshape((48, 48))
                
                data_X.append(image)
                data_y.append(emotion)
                
                if (i + 1) % 1000 == 0:
                    print(f"Loaded {i} images...")
        
        self.X_train = np.array(data_X)
        self.y_train = np.array(data_y)
        
        # Normalize
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        
        print(f"Dataset loaded: {self.X_train.shape}")
        return self.X_train, self.y_train
    
    def load_custom_data(self, data_dir):
        """
        Load custom dataset from directory structure:
        data_dir/
            ├── angry/
            ├── happy/
            ├── sad/
            └── ...
        
        Args:
            data_dir: Root directory containing emotion folders
        """
        print(f"Loading custom dataset from {data_dir}...")
        
        data_X = []
        data_y = []
        emotion_map = {emotion: idx for idx, emotion in enumerate(self.recognizer.emotions)}
        
        for emotion_folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, emotion_folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            emotion_idx = emotion_map.get(emotion_folder.lower())
            if emotion_idx is None:
                print(f"Skipping unknown emotion folder: {emotion_folder}")
                continue
            
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize to 48x48
                    img = cv2.resize(img, (48, 48))
                    
                    data_X.append(img)
                    data_y.append(emotion_idx)
                
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        self.X_train = np.array(data_X, dtype='float32') / 255.0
        self.y_train = np.array(data_y)
        
        # Add channel dimension
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        
        print(f"Custom dataset loaded: {self.X_train.shape}")
        return self.X_train, self.y_train
    
    def split_data(self, test_size=0.2, validation_size=0.2):
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion of test set
            validation_size: Proportion of validation set (from training data)
        """
        if self.X_train is None:
            raise ValueError("No data loaded. Use load_fer2013_data() or load_custom_data() first")
        
        # Split into train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X_train, self.y_train,
            test_size=test_size,
            random_state=42,
            stratify=self.y_train
        )
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size,
            random_state=42,
            stratify=y_temp
        )
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, 7)
        y_val_cat = to_categorical(y_val, 7)
        self.y_test_cat = to_categorical(self.y_test, 7)
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = self.X_test
        self.y_train = y_train_cat
        self.y_val = y_val_cat
        
        print(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
    
    def train(self, epochs=50, batch_size=64, model_save_path='emotion_model.h5'):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_save_path: Path to save trained model
        """
        if self.X_train is None:
            raise ValueError("No data loaded")
        
        # Data augmentation for better generalization
        data_gen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Train the model
        history = self.recognizer.model.fit(
            data_gen.flow(self.X_train, self.y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            steps_per_epoch=len(self.X_train) // batch_size,
            verbose=1
        )
        
        # Save model
        self.recognizer.model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Evaluate on test set
        test_loss, test_acc = self.recognizer.model.evaluate(
            self.X_test, self.y_test_cat,
            verbose=0
        )
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return history


def main():
    """Example training script"""
    
    # Option 1: Load FER2013 dataset
    # trainer = ModelTrainer()
    # trainer.load_fer2013_data('path/to/fer2013.csv')
    
    # Option 2: Load custom dataset
    trainer = ModelTrainer()
    # Create custom dataset directory structure or modify path
    data_dir = 'dataset'  # Should contain emotion folders
    
    if os.path.exists(data_dir):
        trainer.load_custom_data(data_dir)
        trainer.split_data()
        trainer.train(epochs=30, batch_size=32)
    else:
        print(f"Dataset not found at {data_dir}")
        print("Please create dataset with structure:")
        print("dataset/")
        print("  ├── angry/")
        print("  ├── disgust/")
        print("  ├── fear/")
        print("  ├── happy/")
        print("  ├── neutral/")
        print("  ├── sad/")
        print("  └── surprise/")


if __name__ == '__main__':
    main()
