import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
from typing import Tuple, List, Dict
import pickle
class ASLDatasetTrainer:
    """
    ASL Alphabet Dataset Trainer using CNN
    Specifically designed for the ASL_Alphabet_Dataset structure
    """
    
    def __init__(self, dataset_path: str = "ASL_Alphabet_Dataset", img_size: Tuple[int, int] = (64, 64)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Configure GPU
        self.configure_gpu()
        
        # Define the expected classes (ASL alphabet + special classes)
        self.expected_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
    
    def configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"🚀 GPU configured successfully! Found {len(gpus)} GPU(s)")
                print(f"   GPU: {gpus[0].name}")
                print("   Your RTX 4090 will accelerate training significantly!")
                
                # Set mixed precision for faster training on RTX cards
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✅ Mixed precision enabled for maximum speed")
                
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                print("Falling back to CPU training")
        else:
            print("⚠️  No GPU found, using CPU")
            # Configure CPU for optimal performance
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the ASL dataset from the folder structure
        Returns: (images, labels)
        """
        print("Loading ASL dataset...")
        
        train_path = os.path.join(self.dataset_path, "asl_alphabet_train")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        images = []
        labels = []
        
        # Get all class folders
        class_folders = [f for f in os.listdir(train_path) 
                        if os.path.isdir(os.path.join(train_path, f))]
        
        print(f"Found {len(class_folders)} classes: {sorted(class_folders)}")
        
        for class_name in sorted(class_folders):
            class_path = os.path.join(train_path, class_name)
            class_images = [f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(class_images)} images for class '{class_name}'...")
            
            for i, img_file in enumerate(class_images):
                if i % 1000 == 0:  # Progress indicator
                    print(f"  Processed {i}/{len(class_images)} images for {class_name}")
                
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    img = cv2.resize(img, self.img_size)
                    
                    # Normalize pixel values
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        print(f"Loaded {len(images)} total images")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_.tolist()
        
        print(f"Classes: {self.class_names}")
        
        return images, labels_encoded
    
    def create_model(self, num_classes: int) -> keras.Model:
        """
        Create a CNN model for ASL alphabet recognition
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, validation_split: float = 0.2, epochs: int = 20, batch_size: int = 64):
        """
        Train the ASL recognition model
        """
        # Load dataset
        X, y = self.load_dataset()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Create model
        num_classes = len(self.class_names)
        self.model = self.create_model(num_classes)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip for sign language
            fill_mode='nearest'
        )
        
        # Callbacks with more frequent saving
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_asl_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            # Save checkpoint every epoch
            keras.callbacks.ModelCheckpoint(
                'asl_model_checkpoint_epoch_{epoch:02d}.h5',
                save_freq='epoch',
                verbose=1
            ),
            # Save progress every 100 batches
            keras.callbacks.ModelCheckpoint(
                'asl_alphabet_model.h5',
                save_freq=100,
                verbose=0
            )
        ]
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and label encoder
        self.save_model()
        
        return history
    
    def save_model(self):
        """Save the trained model and label encoder"""
        if self.model is not None:
            self.model.save('asl_alphabet_model.h5')
            
            # Save label encoder
            with open('asl_label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save class names
            with open('asl_class_names.json', 'w') as f:
                json.dump(self.class_names, f)
            
            print("Model saved successfully!")
    
    def load_model(self):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model('asl_alphabet_model.h5')
            
            with open('asl_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open('asl_class_names.json', 'r') as f:
                self.class_names = json.load(f)
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASL sign from image
        """
        if self.model is None:
            return "Model not loaded", 0.0
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.resize(image, self.img_size)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, float(confidence)
    
    def evaluate_model(self):
        """Evaluate model on test data if available"""
        test_path = os.path.join(self.dataset_path, "asl_alphabet_test")
        
        if not os.path.exists(test_path):
            print("No test data found")
            return
        
        if self.model is None:
            print("Model not loaded")
            return
        
        # Load test images
        test_images = []
        test_labels = []
        
        test_files = [f for f in os.listdir(test_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for test_file in test_files:
            # Extract label from filename (e.g., "A_test.jpg" -> "A")
            label = test_file.split('_')[0]
            if label == "nothing":
                label = "nothing"
            elif label == "space":
                label = "space"
            
            img_path = os.path.join(test_path, test_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                
                test_images.append(img)
                test_labels.append(label)
        
        test_images = np.array(test_images)
        
        # Make predictions
        predictions = self.model.predict(test_images)
        predicted_classes = [self.class_names[np.argmax(pred)] for pred in predictions]
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(test_labels, predicted_classes) if true == pred)
        accuracy = correct / len(test_labels)
        
        print(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(test_labels)})")
        
        # Show some predictions
        print("\nSample predictions:")
        for i in range(min(10, len(test_labels))):
            confidence = np.max(predictions[i])
            print(f"True: {test_labels[i]}, Predicted: {predicted_classes[i]}, Confidence: {confidence:.4f}")

def main():
    """Main training function"""
    trainer = ASLDatasetTrainer()
    
    print("Starting ASL Alphabet Dataset Training...")
    
    # Train the model
    history = trainer.train_model(epochs=25, batch_size=64)
    
    # Evaluate on test data
    trainer.evaluate_model()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
