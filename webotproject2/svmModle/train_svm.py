# train_svm.py
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "svmModel.joblib"   # output model filename


def load_data(csv_path):
    X = []
    y = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = []
            for key in row.keys():
                # extract features: f0, f1, ..., f41
                if key.startswith("f"):
                    features.append(float(row[key]))
            label = int(row["label"])
            X.append(features)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[INFO] Loaded data from {csv_path}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def main():
    X, y = load_data(DATA_FILE)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardization + SVM (RBF kernel)
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale"
        ))
    ])

    print("[INFO] Training SVM model...")
    svm_clf.fit(X_train, y_train)

    print("[INFO] Evaluating on test set...")
    y_pred = svm_clf.predict(X_test)

    print("\n[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save trained model
    joblib.dump(svm_clf, MODEL_FILE)
    print(f"\n[INFO] Model saved as: {MODEL_FILE}")


if __name__ == "__main__":
    main()
