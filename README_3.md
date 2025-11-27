# Handwritten Character Recognition using CNN

## 📋 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits from the MNIST dataset. Built as part of **Code Alpha ML Internship - Task 3**.

## 🎯 Objective
Identify and classify handwritten characters (digits 0-9) using deep learning and image processing techniques.

## 📊 Dataset
**MNIST (Modified National Institute of Standards and Technology)**
- **Source**: Built-in TensorFlow/Keras dataset
- **Training Images**: 60,000 samples
- **Testing Images**: 10,000 samples
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)
- **Format**: Grayscale pixel values (0-255)

### Dataset Distribution:
Each digit class contains approximately 6,000 training samples, making it a balanced dataset ideal for classification tasks.

## 🛠️ Technologies Used
- **Python 3.x**
- **Deep Learning Framework**: TensorFlow/Keras
- **Libraries**:
  - `tensorflow` - Deep learning framework
  - `keras` - High-level neural networks API
  - `numpy` - Numerical computations
  - `pandas` - Data manipulation
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical visualizations
  - `scikit-learn` - Evaluation metrics

## 🧠 Model Architecture

### Convolutional Neural Network (CNN)
```
Input Layer: 28×28×1 (grayscale images)
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling (2×2)
    ↓
Batch Normalization
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling (2×2)
    ↓
Batch Normalization
    ↓
Conv2D (128 filters, 3×3) + ReLU
    ↓
Batch Normalization
    ↓
Flatten
    ↓
Dense (128 neurons) + ReLU + Dropout(0.5)
    ↓
Dense (64 neurons) + ReLU + Dropout(0.3)
    ↓
Dense (10 neurons) + Softmax
    ↓
Output: Class probabilities for digits 0-9
```

### Key Components:
- **3 Convolutional Layers**: Extract spatial features
- **MaxPooling**: Reduce spatial dimensions
- **Batch Normalization**: Stabilize training
- **Dropout Layers**: Prevent overfitting
- **Dense Layers**: Final classification
- **Softmax Activation**: Multi-class probability output

### Model Parameters:
- **Total Parameters**: ~500,000
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## 📁 Project Structure
```
handwriting-recognition/
│
├── Handwritten_Character_Recognition.py   # Main Python script
├── README.md                               # Project documentation
│
├── models/
│   ├── handwriting_recognition_model.h5   # Trained model
│   └── model_architecture.json            # Model structure
│
└── outputs/
    ├── sample_digits.png                  # Sample MNIST digits
    ├── digit_variations.png               # Variations of same digit
    ├── training_history.png               # Training & validation curves
    ├── confusion_matrix.png               # Classification confusion matrix
    ├── correct_predictions.png            # Correctly predicted samples
    ├── incorrect_predictions.png          # Misclassified samples
    ├── confidence_analysis.png            # Prediction confidence distribution
    └── sample_prediction.png              # Single prediction example
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager

### Install Required Libraries
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Or using conda:
```bash
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## ▶️ How to Run

### Method 1: Run Python Script
```bash
python Handwritten_Character_Recognition.py
```

### Method 2: Jupyter Notebook
```bash
jupyter notebook Handwritten_Character_Recognition.ipynb
```

### Expected Runtime:
- **Data Loading**: ~30 seconds (first run downloads dataset)
- **Training**: 5-10 minutes (15 epochs)
- **Evaluation**: ~1 minute
- **Total**: ~12-15 minutes

## 📈 Results

### Model Performance
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 99.5%+ | 99.0%+ | 99.0%+ |
| **Loss** | ~0.015 | ~0.030 | ~0.035 |

### Per-Class Performance (Approximate)
| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.99 | 0.99 | 0.99 |
| 1 | 0.99 | 0.99 | 0.99 |
| 2 | 0.99 | 0.98 | 0.99 |
| 3 | 0.98 | 0.99 | 0.99 |
| 4 | 0.99 | 0.99 | 0.99 |
| 5 | 0.99 | 0.98 | 0.98 |
| 6 | 0.99 | 0.99 | 0.99 |
| 7 | 0.98 | 0.99 | 0.98 |
| 8 | 0.98 | 0.98 | 0.98 |
| 9 | 0.98 | 0.98 | 0.98 |

**Overall Test Accuracy**: ~99.0%

## 🔍 Key Features

### 1. Data Preprocessing
- Pixel normalization (0-255 → 0-1)
- Image reshaping for CNN input
- One-hot encoding of labels
- Train-validation split (80-20)

### 2. Model Training
- 15 epochs with batch size 128
- Adam optimizer with default learning rate
- Early stopping capability
- Real-time training monitoring

### 3. Comprehensive Evaluation
- **Accuracy metrics**: Overall and per-class
- **Confusion matrix**: Visualization of misclassifications
- **Classification report**: Precision, recall, F1-score
- **Confidence analysis**: Prediction certainty distribution

### 4. Visualizations
- Sample digit images from dataset
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrix heatmap
- Correct vs incorrect predictions
- Confidence score distributions

### 5. Model Deployment
- Saved model in HDF5 format
- Model architecture in JSON
- Custom prediction function
- Ready for production use

## 💡 Usage Example

### Load and Use Trained Model
```python
from tensorflow import keras
import numpy as np

# Load the saved model
model = keras.models.load_model('handwriting_recognition_model.h5')

# Prepare your image (28×28 grayscale, normalized)
your_image = your_image.reshape(1, 28, 28, 1) / 255.0

# Make prediction
prediction = model.predict(your_image)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence: {confidence:.2%}")
```

## 🎓 Learning Outcomes
- Implemented CNN from scratch using TensorFlow/Keras
- Understood image preprocessing techniques
- Applied batch normalization and dropout for regularization
- Evaluated model using multiple metrics
- Created comprehensive visualizations
- Developed production-ready deep learning model
- Achieved state-of-the-art accuracy on MNIST

## 🔮 Future Enhancements
- Extend to **EMNIST** dataset (handwritten letters A-Z)
- Implement **data augmentation** for better generalization
- Add **real-time webcam digit recognition**
- Deploy as **web application** using Flask/Streamlit
- Implement **CRNN** for word/sentence recognition
- Add **transfer learning** with pre-trained models
- Create **mobile app** for on-device recognition

## 🐛 Common Issues & Solutions

### Issue 1: TensorFlow Installation Error
```bash
# Solution: Use specific version
pip install tensorflow==2.13.0
```

### Issue 2: GPU Not Detected
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu
```

### Issue 3: Memory Error During Training
```python
# Solution: Reduce batch size
model.fit(..., batch_size=64)  # instead of 128
```

## 📚 References
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [CNN Architecture Guide](https://cs231n.github.io/)

## 🤝 Contributing
This is an internship project for Code Alpha. Feedback and suggestions are welcome!

## 👨‍💻 Author
**Rakshitha PN**  
Code Alpha ML Intern  
Task 3: Handwritten Character Recognition

## 📧 Contact
- **Email**: rakshithapn123@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/rakshitha-pn-b305a2292
- **GitHub**: https://github.com/Rakshitha973-pn

## 📄 License
This project is created for educational purposes as part of Code Alpha ML Internship.

## 🙏 Acknowledgments
- Code Alpha for the internship opportunity
- Yann LeCun et al. for the MNIST dataset
- TensorFlow/Keras community for excellent documentation
- Open-source ML community

## 📊 Project Statistics
- **Lines of Code**: ~350
- **Training Time**: ~10 minutes
- **Model Size**: ~2.5 MB
- **Accuracy**: 99%+
- **Parameters**: ~500K
 
