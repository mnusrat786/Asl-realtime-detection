import cv2
import numpy as np
import tensorflow as tf
import json
import os

# Disable GPU to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_prediction():
    print("🤟 ASL Prediction Test (CPU Only)")
    print("=" * 40)
    
    try:
        # Load model
        print("📚 Loading model...")
        model = tf.keras.models.load_model('asl_alphabet_model.h5')
        
        with open('asl_class_names.json', 'r') as f:
            class_names = json.load(f)
        
        print(f"✅ Model loaded! {len(class_names)} classes")
        
        # Test with dataset image first
        test_path = "ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg"
        print(f"📸 Testing with: {test_path}")
        
        # Load and preprocess
        image = cv2.imread(test_path)
        processed = cv2.resize(image, (64, 64))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = processed.astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        print("🔍 Making prediction (CPU only)...")
        
        # Use __call__ instead of predict to avoid issues
        predictions = model(processed, training=False)
        predictions = predictions.numpy()
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_sign = class_names[predicted_idx]
        
        print("\n" + "="*40)
        print(f"🎯 PREDICTED: {predicted_sign}")
        print(f"📊 CONFIDENCE: {confidence:.3f}")
        print(f"🎯 EXPECTED: A")
        
        if predicted_sign == 'A':
            print("🎉 PERFECT! Model working correctly!")
        else:
            print("⚠️  Different result, but model is running")
        
        print("="*40)
        
        # Show top 3
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        print("\n📋 Top 3 predictions:")
        for i, idx in enumerate(top_indices, 1):
            sign = class_names[idx]
            conf = predictions[0][idx]
            print(f"{i}. {sign}: {conf:.3f}")
        
        # Now test Screenshot2025.png
        print(f"\n📸 Testing Screenshot2025.png...")
        
        screenshot = cv2.imread('Screenshot2025.png')
        if screenshot is not None:
            processed_screenshot = cv2.resize(screenshot, (64, 64))
            processed_screenshot = cv2.cvtColor(processed_screenshot, cv2.COLOR_BGR2RGB)
            processed_screenshot = processed_screenshot.astype(np.float32) / 255.0
            processed_screenshot = np.expand_dims(processed_screenshot, axis=0)
            
            screenshot_predictions = model(processed_screenshot, training=False)
            screenshot_predictions = screenshot_predictions.numpy()
            
            screenshot_idx = np.argmax(screenshot_predictions[0])
            screenshot_confidence = float(screenshot_predictions[0][screenshot_idx])
            screenshot_sign = class_names[screenshot_idx]
            
            print(f"\n🖼️  YOUR SCREENSHOT RESULT:")
            print(f"🎯 DETECTED: {screenshot_sign}")
            print(f"📊 CONFIDENCE: {screenshot_confidence:.3f}")
            
            if screenshot_confidence > 0.6:
                print("🎉 HIGH CONFIDENCE!")
            elif screenshot_confidence > 0.3:
                print("🤔 MEDIUM CONFIDENCE")
            else:
                print("😕 LOW CONFIDENCE")
            
            # Show top 3 for screenshot
            top_screenshot = np.argsort(screenshot_predictions[0])[-3:][::-1]
            print("\n📋 Top 3 for your image:")
            for i, idx in enumerate(top_screenshot, 1):
                sign = class_names[idx]
                conf = screenshot_predictions[0][idx]
                print(f"{i}. {sign}: {conf:.3f}")
        
        print(f"\n✅ SUCCESS! ASL detection is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()