# collect_svm_data.py
import cv2
import csv
import os
import mediapipe as mp
from extract_features import extract_hand_features

# =========================================================
# 1. Set dataset directory (auto-create)
# =========================================================
DATA_DIR = r"E:\webot_project\webotproject2\svmModle\TEST"
os.makedirs(DATA_DIR, exist_ok=True)

# Final CSV save path
DATA_FILE = os.path.join(DATA_DIR, "gesture_test_data.csv")

# Label descriptions (for display only; training uses numeric labels)
LABEL_NAMES = {
    0: "FIST (Emergency Stop)",
    1: "PALM_FORWARD (Palm facing camera, fingers up: Forward)",
    2: "PALM_RIGHT (Palm facing camera, fingers pointing right: Turn Right)",
    3: "PALM_LEFT (Back of hand facing camera, fingers pointing left: Turn Left)",
    4: "ONE (1 finger extended: Speed Up)",
    5: "TWO (2 fingers extended: Slow Down)",
}

def main():
    # Create file and write header if not exists
    file_exists = os.path.exists(DATA_FILE)
    f = open(DATA_FILE, "a", newline="")
    writer = csv.writer(f)

    if not file_exists:
        # 42-dimensional features: f0 ... f41 + label
        header = [f"f{i}" for i in range(42)] + ["label"]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[Collect] Failed to open camera.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    current_label = 0
    saved_count = 0

    print("[Collect] Camera initialized.")
    print("[Collect] Instructions:")
    print("  Press number keys 0–5 to switch gesture class.")
    print("  Press 's' to save current frame to dataset.")
    print("  Press 'q' or ESC to exit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = hands.process(img_rgb)
                img_rgb.flags.writeable = True

                h, w, _ = frame.shape

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                # Show current label on screen
                cv2.putText(
                    frame,
                    f"Label: {current_label} - {LABEL_NAMES[current_label]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Saved samples: {saved_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("Collect SVM Gesture Data", frame)
                key = cv2.waitKey(1) & 0xFF

                # Switch label 0–5
                if key in [ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
                    current_label = int(chr(key))
                    print(f"[Collect] Switched label to {current_label} - {LABEL_NAMES[current_label]}")

                # Save one frame
                if key == ord("s"):
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        features = extract_hand_features(hand_landmarks.landmark)
                        if features.shape[0] != 42:
                            print("[Collect] Feature dimension is not 42. Check extract_hand_features.")
                        else:
                            row = list(features) + [current_label]
                            writer.writerow(row)
                            f.flush()
                            saved_count += 1
                            print(f"[Collect] Saved sample #{saved_count}, label {current_label}")
                    else:
                        print("[Collect] No hand detected in this frame, not saved.")

                # Exit program
                if key == ord("q") or key == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            f.close()
            print("[Collect] Data collection finished. File saved to:", DATA_FILE)

if __name__ == "__main__":
    main()
