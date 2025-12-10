import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ======== Class label mapping (numeric → gesture name) ========
LABEL_NAME_MAP = {
    0: "FIST / Stop",
    1: "Palm Up / Forward",
    2: "Palm Right / TurnR",
    3: "Palm Left / TurnL",
    4: "One Finger / Speed+",
    5: "Two Fingers / Speed-",
}

# ======== Path settings ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svmModel.joblib")
TEST_CSV = os.path.join(BASE_DIR, "gesture_test_data.csv")  # Test dataset

print("Loading model from:", MODEL_PATH)
clf = joblib.load(MODEL_PATH)

print("Loading test data from:", TEST_CSV)
df = pd.read_csv(TEST_CSV)

# ======== Data preparation ========
y_true = df["label"].values
X = df.drop(columns=["label"]).values

y_pred = clf.predict(X)

# All labels present in the dataset
labels = sorted(np.unique(y_true).tolist())

# Display names for visualization
display_names = [LABEL_NAME_MAP[l] for l in labels]

# ======== Generate Confusion Matrix ========
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nConfusion Matrix (Rows = True, Columns = Pred):\n")
print(cm)

# ======== Plot Confusion Matrix Heatmap ========
plt.figure(figsize=(10, 7))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_names,
            yticklabels=display_names)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("SVM Confusion Matrix for Gesture Classification")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
