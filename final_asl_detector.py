import cv2
import numpy as np
import tensorflow as tf
import json
import os
import time

# Force CPU usage to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PersistentASLDetector:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.load_model()
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.process_every_n_frames = 5
        
    def load_model(self):
        """Load the ASL model"""
        try:
            print("📚 Loading ASL model...")
            self.model = tf.keras.models.load_model('asl_alphabet_model.h5')
            
            with open('asl_class_names.json', 'r') as f:
                self.class_names = json.load(f)
            
            print(f"✅ Model loaded! {len(self.class_names)} signs ready")
            return True
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
    
    def setup_camera_persistent(self):
        """Setup camera that won't quit unexpectedly"""
        print("🎥 Setting up persistent camera...")
        
        # Try DirectShow first (usually most stable on Windows)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("   DirectShow failed, trying default...")
            cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            # Set properties for stability
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
            
            # Test camera
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Camera ready and persistent!")
                return cap
        
        print("❌ Camera setup failed")
        return None
    
    def predict_asl_safe(self, image_region):
        """Safe ASL prediction"""
        try:
            # Preprocess
            processed = cv2.resize(image_region, (64, 64))
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            # Predict using the working method
            predictions = self.model(processed, training=False)
            predictions = predictions.numpy()
            
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_sign = self.class_names[predicted_idx]
            
            return predicted_sign, confidence
            
        except Exception as e:
            return "Error", 0.0
    
    def run_persistent_detection(self):
        """Run detection that won't auto-quit"""
        if self.model is None:
            print("❌ Cannot run without model")
            return
        
        cap = self.setup_camera_persistent()
        if cap is None:
            print("❌ Cannot run without camera")
            return
        
        print("🎥 PERSISTENT ASL Detection Started!")
        print("=" * 50)
        print("🔄 This will run continuously until YOU quit")
        print("📋 Controls:")
        print("   - Put your hand in the GREEN BOX")
        print("   - Make ASL signs (A, B, C, L, N, etc.)")
        print("   - Press 'q' to quit (ONLY way to stop)")
        print("   - Press '+/-' to adjust confidence")
        print("   - Press 's' to save frame")
        print("=" * 50)
        
        frame_count = 0
        last_prediction = "Ready..."
        last_confidence = 0.0
        consecutive_errors = 0
        max_consecutive_errors = 50  # Allow more errors before giving up
        
        print("🚀 Starting main loop...")
        
        while True:  # Infinite loop - only exits on 'q' press
            try:
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_errors += 1
                    print(f"⚠️ Frame read error {consecutive_errors}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("❌ Too many consecutive errors, restarting camera...")
                        cap.release()
                        time.sleep(2)
                        cap = self.setup_camera_persistent()
                        if cap is None:
                            print("❌ Camera restart failed")
                            break
                        consecutive_errors = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Define detection box (center of screen)
                box_size = 200
                x1 = w//2 - box_size//2
                y1 = h//2 - box_size//2
                x2 = x1 + box_size
                y2 = y1 + box_size
                
                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, "ASL SIGN HERE", (x1-20, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Process every Nth frame for performance
                if frame_count % self.process_every_n_frames == 0:
                    hand_region = frame[y1:y2, x1:x2]
                    if hand_region.size > 0:
                        prediction, confidence = self.predict_asl_safe(hand_region)
                        last_prediction = prediction
                        last_confidence = confidence
                
                # Display results
                if last_confidence > self.confidence_threshold:
                    color = (0, 255, 0)  # Green - high confidence
                    status = "DETECTED"
                elif last_confidence > 0.3:
                    color = (0, 165, 255)  # Orange - medium confidence
                    status = "MAYBE"
                else:
                    color = (0, 0, 255)  # Red - low confidence
                    status = "TRYING"
                
                # Draw result background
                cv2.rectangle(frame, (10, 10), (600, 120), (0, 0, 0), -1)
                
                # Main result
                cv2.putText(frame, f"{status}: {last_prediction}", (15, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Confidence
                cv2.putText(frame, f"Confidence: {last_confidence:.3f}", (15, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Settings
                cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f} | Frame: {frame_count}", 
                           (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Instructions at bottom
                cv2.putText(frame, "Press 'q' to QUIT | Make ASL signs in green box", 
                           (15, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Persistent ASL Detection', frame)
                
                # Handle key presses - ONLY 'q' will quit
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 User pressed 'q' - Quitting...")
                    break
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.1)
                    print(f"📈 Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                    print(f"📉 Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('s'):
                    filename = f"asl_capture_{frame_count}_{last_prediction}_{last_confidence:.2f}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                elif key != 255:  # Any other key pressed
                    print(f"ℹ️ Key pressed: {key} (only 'q' will quit)")
                
            except KeyboardInterrupt:
                print("\n👋 Ctrl+C pressed - Quitting...")
                break
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                # Don't quit on errors, just continue
                time.sleep(0.1)
                continue
        
        # Cleanup
        print("🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def main():
    print("🤟 PERSISTENT ASL Live Detector")
    print("=" * 50)
    print("🔄 This detector will NOT auto-quit!")
    print("🛑 Only stops when YOU press 'q'")
    print("🚀 Starting detector...")
    
    detector = PersistentASLDetector()
    detector.run_persistent_detection()
    
    print("👋 Detector ended")

if __name__ == "__main__":
    main()