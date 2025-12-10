import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ======== Class Label Mapping (numeric → gesture name for visualization) ========
# You can modify the names as needed
LABEL_NAME_MAP = {
    0: "FIST / Stop",
    1: "Palm Up / Forward",
    2: "Palm Right / TurnR",
    3: "Palm Left / TurnL",
    4: "One Finger / Speed+",
    5: "Two Fingers / Speed-",
}

# ======== Load Model and Test Data ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # the TEST directory
MODEL_PATH = os.path.join(BASE_DIR, "svmModel.joblib")
TEST_CSV = os.path.join(BASE_DIR, "gesture_test_data.csv")   # test dataset

print("Loading model from:", MODEL_PATH)
clf = joblib.load(MODEL_PATH)

print("Loading test data from:", TEST_CSV)
df = pd.read_csv(TEST_CSV)

# y_true uses numeric labels (0–5)
y_true = df["label"].values
X = df.drop(columns=["label"]).values

# Predict
y_pred = clf.predict(X)

# Get all unique labels present in the dataset
labels = sorted(np.unique(y_true).tolist())

# Gesture names for visualization (same order as labels)
display_names = [LABEL_NAME_MAP.get(l, str(l)) for l in labels]

# ======== Confusion Matrix ========
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("\nConfusion Matrix (rows = true label, cols = predicted label):\n", cm)

# ======== Classification Report (Precision / Recall / F1) ========
print("\n=== Classification Report (per class) ===")
print(classification_report(y_true, y_pred, target_names=display_names))

# ======== Extract per-class metrics (for visualization) ========
report = classification_report(
    y_true, y_pred,
    target_names=display_names,
    output_dict=True
)

precisions = []   # per-class precision
recalls = []
f1_scores = []

for name in display_names:
    precisions.append(report[name]["precision"])
    recalls.append(report[name]["recall"])
    f1_scores.append(report[name]["f1-score"])

# ======== Visualization: Precision / Recall / F1 ========
plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.25

plt.bar(x - width, precisions, width, label="Precision")
plt.bar(x,         recalls,    width, label="Recall")
plt.bar(x + width, f1_scores,  width, label="F1-score")

plt.xticks(x, display_names, rotation=20, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.xlabel("Gesture Class")
plt.title("SVM Performance per Gesture Class")
plt.legend()
