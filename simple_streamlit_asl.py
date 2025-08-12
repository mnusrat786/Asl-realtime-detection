import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from PIL import Image

# Page config
st.set_page_config(page_title="ASL Detector", page_icon="🤟", layout="wide")

@st.cache_resource
def load_model():
    """Load ASL model"""
    try:
        model = keras.models.load_model('asl_alphabet_model.h5')
        with open('asl_class_names.json', 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

def predict_asl(image, model, class_names):
    """Predict ASL sign from image"""
    if model is None:
        return "Model not loaded", 0.0
    
    # Convert image to numpy array if it's PIL
    if hasattr(image, 'mode'):
        image = np.array(image)
    
    # Handle different image formats
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # Already RGB
            pass
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Preprocess image
    processed = cv2.resize(image, (64, 64))
    processed = processed.astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=0)
    
    # Predict using CPU to avoid CUDA issues
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    try:
        # Use model call instead of predict to avoid issues
        predictions = model(processed, training=False)
        predictions = predictions.numpy()
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_sign = class_names[predicted_idx]
        
        return predicted_sign, confidence
    except Exception as e:
        return f"Prediction error: {str(e)}", 0.0

def main():
    st.title("🤟 ASL Sign Language Detector")
    st.markdown("Upload an image or use your camera to detect ASL signs!")
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        st.error("❌ Cannot load ASL model!")
        st.stop()
    
    st.success(f"✅ Model loaded! Can detect {len(class_names)} signs")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📸 Upload Image", "🎥 Camera (Instructions)"])
    
    with tab1:
        st.subheader("Upload an image of your ASL sign")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of your hand making an ASL sign"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("🎯 Detection Results")
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    prediction, confidence = predict_asl(image_np, model, class_names)
                
                # Display results
                if confidence > 0.6:
                    st.success(f"**DETECTED: {prediction}**")
                    st.metric("Confidence", f"{confidence:.3f}")
                elif confidence > 0.3:
                    st.warning(f"**MAYBE: {prediction}**")
                    st.metric("Confidence", f"{confidence:.3f}")
                else:
                    st.error(f"**LOW CONFIDENCE: {prediction}**")
                    st.metric("Confidence", f"{confidence:.3f}")
                
                # Show all predictions
                with st.expander("See all predictions"):
                    processed = cv2.resize(image_np, (64, 64))
                    processed = processed.astype(np.float32) / 255.0
                    processed = np.expand_dims(processed, axis=0)
                    
                    all_predictions = model.predict(processed, verbose=0)[0]
                    
                    # Create dataframe of top 5 predictions
                    import pandas as pd
                    
                    top_indices = np.argsort(all_predictions)[-5:][::-1]
                    top_predictions = []
                    
                    for idx in top_indices:
                        top_predictions.append({
                            'Sign': class_names[idx],
                            'Confidence': f"{all_predictions[idx]:.3f}"
                        })
                    
                    df = pd.DataFrame(top_predictions)
                    st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("🎥 Camera Instructions")
        st.markdown("""
        Since the camera integration has issues, here's how to use ASL detection:
        
        ### 📱 **Method 1: Take a photo with your phone**
        1. Open your phone camera
        2. Make an ASL sign clearly
        3. Take a photo
        4. Upload it in the "Upload Image" tab
        
        ### 💻 **Method 2: Use Windows Camera app**
        1. Open Windows Camera app
        2. Make an ASL sign
        3. Take a photo
        4. Save and upload here
        
        ### 🤟 **Tips for better detection:**
        - **Good lighting** - make sure your hand is well-lit
        - **Plain background** - use a wall or plain surface
        - **Clear hand position** - hold the sign steady
        - **Fill the frame** - make your hand the main subject
        
        ### 📚 **ASL Signs to try:**
        """)
        
        # Show some example signs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Letter A**: Closed fist, thumb on side")
            st.markdown("**Letter B**: Four fingers up, thumb tucked")
            st.markdown("**Letter C**: Make a 'C' shape")
        
        with col2:
            st.markdown("**Letter L**: Index finger up, thumb out")
            st.markdown("**Letter O**: Make an 'O' shape")
            st.markdown("**Letter Y**: Thumb and pinky out")
        
        with col3:
            st.markdown("**Letter I**: Pinky finger up")
            st.markdown("**Letter V**: Peace sign (2 fingers)")
            st.markdown("**Letter W**: Three fingers up")
        
        st.info("💡 The model can detect all 26 letters plus 'del', 'nothing', and 'space'")

if __name__ == "__main__":
    main()