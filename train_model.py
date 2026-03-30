import os
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import skew, kurtosis
from scipy.fftpack import fft


train_path = os.path.join("datasets", "ecg data old version", "ecg data old version", "train")
model_dir = "model"
csv_path = "dataset.csv"

os.makedirs(model_dir, exist_ok=True)


category_map = {
    "Normal Person ECG Images (284x12=3408)": 0,
    "ECG Images of Myocardial Infarction Patients (240x12=2880)": 1,
    "ECG Images of Patient that have History of MI (172x12=2064)": 2,
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": 3
}


with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])

    for folder, label in category_map.items():

        full_folder = os.path.join(train_path, folder)

        if not os.path.exists(full_folder):
            continue

        for img in os.listdir(full_folder):

            if img.lower().endswith((".jpg", ".png", ".jpeg")):

                writer.writerow([os.path.join(full_folder, img), label])


df = pd.read_csv(csv_path)


features = []
labels = []

img_size = 128


for _, row in df.iterrows():

    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0

    flat = img.flatten()

    if np.std(flat) < 1e-6:
        continue

    mean_val = np.mean(flat)
    std_val = np.std(flat)
    skew_val = skew(flat)
    kurt_val = kurtosis(flat)

    if not np.isfinite(skew_val) or not np.isfinite(kurt_val):
        continue

    fft_vals = np.abs(fft(flat))[:50]

    fv = [mean_val, std_val, skew_val, kurt_val]
    fv.extend(fft_vals)

    features.append(fv)
    labels.append(row["label"])


X = np.array(features, dtype=np.float32)
y = np.array(labels)


scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
all_y_true = []
all_y_pred = []
all_y_prob = []

history = None


for train_idx, test_idx in kfold.split(X, y):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_train_cat = to_categorical(y_train, num_classes=4)
    y_test_cat = to_categorical(y_test, num_classes=4)

    model = Sequential([
        Dense(128, activation="relu", input_dim=X.shape[1]),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train_cat,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stop],
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)

    accuracies.append(acc)

    y_prob = model.predict(X_test)

    y_pred = np.argmax(y_prob, axis=1)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_prob.extend(y_prob)


avg_accuracy = np.mean(accuracies) * 100
std_accuracy = np.std(accuracies) * 100


print("\n10-Fold Cross Validation Results:")

print(f"Average Accuracy: {avg_accuracy:.2f}%")
print(f"Standard Deviation: {std_accuracy:.2f}")


report = classification_report(

    all_y_true,
    all_y_pred,

    target_names=[

        "Normal",
        "Myocardial Infarction",
        "History of MI",
        "Abnormal Heartbeat"

    ],

    output_dict=True,
    zero_division=0

)


report_df = pd.DataFrame(report).transpose()


if "accuracy" in report_df.index:

    report_df = report_df.drop(["accuracy"])


for col in ["precision", "recall", "f1-score"]:

    report_df[col] = (report_df[col] * 100).round(2)


report_df["support"] = report_df["support"].astype(int)


print("\nClassification Report (in %):\n")

print(report_df)



cm = confusion_matrix(all_y_true, all_y_pred)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[

        "Normal",
        "Myocardial Infarction",
        "History of MI",
        "Abnormal Heartbeat"

    ]

)


plt.figure()

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()



plt.figure()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.figure()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()



y_true_bin = label_binarize(all_y_true, classes=[0,1,2,3])

y_pred_prob = np.array(all_y_prob)


plt.figure()


class_names = [

    "Normal",
    "Myocardial Infarction",
    "History of MI",
    "Abnormal Heartbeat"

]


for i in range(4):

    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')


plt.plot([0,1],[0,1],'k--')

plt.title('ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()



model.save(os.path.join(model_dir, "ann_model.h5"))

joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))