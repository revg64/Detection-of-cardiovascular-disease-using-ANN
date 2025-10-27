import os
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import seaborn as sns
import joblib   # save scaler

# ----------------------------
# Step 1: Dataset Preparation
# ----------------------------
base_path = r"C:\Users\Kiran\OneDrive\Desktop\project\datasets"

with open("dataset.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])

    for label, category in enumerate(["normal", "mi", "hmi", "haveingabnormalheartbeat"]):
        folder = os.path.join(base_path, category)
        print("Looking inside:", folder)

        if not os.path.exists(folder):
            print("Folder missing:", folder)
            continue

        for image_file in os.listdir(folder):
            if image_file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder, image_file)
                writer.writerow([img_path, label])

print(" CSV file created successfully: dataset.csv")

# ----------------------------
# Step 2: Feature Extraction
# ----------------------------
df = pd.read_csv("dataset.csv")
features = []
labels = []

img_size = 64
fft_size = 200   # increased FFT components for richer features

for index, row in df.iterrows():
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    flat = img.flatten()
    
    # statistical features
    mean_val = np.mean(flat)
    std_val = np.std(flat)
    skew_val = skew(flat)
    kurt_val = kurtosis(flat)
    
    # frequency features
    fft_vals = np.abs(fft(flat))[:fft_size]
    
    feature_vector = [mean_val, std_val, skew_val, kurt_val]
    feature_vector.extend(fft_vals)
    
    features.append(feature_vector)
    labels.append(row["label"])

X = np.array(features)
y = to_categorical(np.array(labels))

# ----------------------------
# Step 3: Feature Scaling
# ----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")  # save scaler

# ----------------------------
# Step 4: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Step 5: ANN Model
# ----------------------------
model = Sequential([
    Dense(512, input_dim=X.shape[1], activation="relu"),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(y.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=1)
]

# ----------------------------
# Step 6: Training
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ----------------------------
# Step 7: Evaluation
# ----------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n ANN Test Accuracy: {acc*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("\n Classification Report:\n")
print(classification_report(
    y_true, y_pred_classes,
    target_names=["normal", "mi", "hmi", "haveingabnormalheartbeat"]
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["normal", "mi", "hmi", "haveingabnormalheartbeat"],
            yticklabels=["normal", "mi", "hmi", "haveingabnormalheartbeat"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Accuracy curve
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# ----------------------------
# Step 8: Save Model
# ----------------------------
model.save("ecg_ann_model.h5")
print(" Model and scaler saved successfully.")
