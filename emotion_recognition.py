"""
Facial Emotion Recognition System for Raspberry Pi
Optimized for low-power edge devices
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class EmotionRecognizer:
    """
    A lightweight emotion recognition system optimized for Raspberry Pi
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognizer
        
        Args:
            model_path: Path to pre-trained model. If None, creates a new lightweight model
        """
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (255, 0, 0),    # Blue
            'Fear': (255, 0, 255),     # Magenta
            'Happy': (0, 255, 0),      # Green
            'Neutral': (255, 255, 0),  # Cyan
            'Sad': (0, 255, 255),      # Yellow
            'Surprise': (255, 165, 0)  # Orange
        }
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load emotion model or create lightweight one
        if model_path and model_path.endswith('.h5'):
            try:
                self.model = load_model(model_path)
                print(f"Loaded model from {model_path}")
            except:
                print("Failed to load model, creating new lightweight model")
                self.model = self.create_lightweight_model()
        else:
            self.model = self.create_lightweight_model()
    
    def create_lightweight_model(self):
        """
        Create a lightweight CNN model optimized for Raspberry Pi
        Input: 48x48 grayscale images
        Output: 7 emotion classes
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_face(self, face):
        """
        Preprocess face image for model prediction
        
        Args:
            face: Face image from detector
            
        Returns:
            Preprocessed face ready for model
        """
        # Convert to grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        face_resized = cv2.resize(face_gray, (48, 48))
        
        # Normalize pixel values
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Reshape for model input
        face_reshaped = np.expand_dims(face_normalized, axis=(0, -1))
        
        return face_reshaped
    
    def predict_emotion(self, face):
        """
        Predict emotion for a face
        
        Args:
            face: Face image
            
        Returns:
            emotion: Predicted emotion label
            confidence: Confidence score
        """
        processed_face = self.preprocess_face(face)
        prediction = self.model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(prediction[0])
        emotion = self.emotions[emotion_idx]
        confidence = prediction[0][emotion_idx]
        
        return emotion, confidence
    
    def detect_and_recognize(self, frame):
        """
        Detect faces and recognize emotions in a frame
        
        Args:
            frame: Video frame
            
        Returns:
            frame_with_annotations: Frame with drawn rectangles and emotion labels
            emotions_detected: List of detected emotions
        """
        emotions_detected = []
        frame_output = frame.copy()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion, confidence = self.predict_emotion(face)
            emotions_detected.append({
                'emotion': emotion,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })
            
            # Draw rectangle and label
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame_output, (x, y), (x+w, y+h), color, 2)
            
            # Add emotion label with confidence
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(
                frame_output,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )
        
        return frame_output, emotions_detected


class RaspberryPiEmotionApp:
    """
    Main application for running emotion recognition on Raspberry Pi
    """
    
    def __init__(self, model_path=None, camera_index=0):
        """
        Initialize the application
        
        Args:
            model_path: Path to pre-trained model
            camera_index: Camera device index (default: 0 for built-in camera)
        """
        self.recognizer = EmotionRecognizer(model_path)
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
    
    def start_camera(self):
        """Start the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set camera properties for better performance on Raspberry Pi
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        print("Camera started successfully")
        return True
    
    def run(self):
        """Run the emotion recognition application"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("Starting emotion recognition... Press 'q' to quit")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error reading frame")
                    break
                
                # Detect and recognize emotions
                frame_output, emotions = self.recognizer.detect_and_recognize(frame)
                
                # Display frame
                cv2.imshow('Emotion Recognition', frame_output)
                
                # Print detected emotions
                if emotions:
                    for detected in emotions:
                        print(f"Emotion: {detected['emotion']}, "
                              f"Confidence: {detected['confidence']:.2f}")
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the application and release resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application stopped")


if __name__ == "__main__":
    # Create and run the application
    app = RaspberryPiEmotionApp()
    app.run()
