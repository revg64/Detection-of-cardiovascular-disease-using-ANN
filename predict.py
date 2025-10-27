import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
import joblib   # to load scaler

# Load model and scaler
model = load_model("ecg_ann_model.h5")
scaler = joblib.load("scaler.pkl")

img_size = 64
categories = ["normal", "mi", "hmi", "haveingabnormalheartbeat"]

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Image not found"
    
    # Resize and normalize
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    flat = img.flatten()
    
    # Statistical features
    mean_val = np.mean(flat)
    std_val = np.std(flat)
    skew_val = skew(flat)
    kurt_val = kurtosis(flat)
    
    # FFT features → use 200 components (to match training)
    fft_vals = np.abs(fft(flat))[:200]
    
    # Combine features
    feature_vector = [mean_val, std_val, skew_val, kurt_val]
    feature_vector.extend(fft_vals)
    
    # Convert to numpy array and scale
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)
    
    # Prediction
    prediction = model.predict(feature_vector)
    class_idx = np.argmax(prediction)
    return categories[class_idx]

# Example usage
print("Prediction:", predict_image(r"C:\Users\Kiran\OneDrive\Desktop\MI(2).jpg"))
