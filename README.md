#  ASL Real-Time Detection System

Real-time American Sign Language alphabet detection using deep learning and computer vision.

# Results



## Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd asl-detection-system
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv hand_tracking_env

# Activate environment
# Windows:
hand_tracking_env\Scripts\activate
# macOS/Linux:
source hand_tracking_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset
Download the ASL Alphabet Dataset and extract to:
```
ASL_Alphabet_Dataset/
├── asl_alphabet_train/
│   ├── A/ (8400+ images)
│   ├── B/ (8400+ images)
│   └── ... (29 folders total)
└── asl_alphabet_test/
    ├── A_test.jpg
    └── ... (29 test images)
```

### 4. Train Model
```bash
python asl_dataset_trainer.py
```

### 5. Run Detection
```bash
python final_asl_detector.py
```

## Features
- **29 ASL signs** recognition (A-Z + del, nothing, space)
- **Real-time detection** using webcam
- **95%+ accuracy** on training data
- **Cross-platform** compatibility

## 🔧 Alternative Usage

### Test with Images
```bash
python working_prediction_test.py
```

### Web Interface
```bash
streamlit run simple_streamlit_asl.py
```

### Camera Test
```bash
python camera_test_cli.py
```

## 📋 Controls
- **'q'**: Quit application
- **'+'**: Increase confidence threshold
- **'-'**: Decrease confidence threshold
- **'s'**: Save current frame

## 🛠️ Troubleshooting

### Camera Issues
```bash
python camera_test_cli.py
```

### CUDA Issues
```bash
pip install tensorflow-cpu
```

## 📊 Model Performance
- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 90-95%
- **Real-time FPS**: 15-30
- **Model Size**: ~4MB

## 🎯 System Requirements
- Python 3.8+
- 8GB RAM (16GB recommended)
- Webcam
- 3GB storage space

---

**Happy Signing! 🤟**
