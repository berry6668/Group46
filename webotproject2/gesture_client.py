# E:\webotproject2\gesture_client.py
import cv2
import socket
import mediapipe as mp
import numpy as np
import joblib

import os

# ========= Model Path =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svmModle", "svmModel.joblib")

print("Loaded SVM model path:", MODEL_PATH)

# Mapping: numerical labels used during training -> gesture names
ML_LABELS = {
    0: "FIST",          # STOP
    1: "PALM_FORWARD",  # FORWARD
    2: "PALM_RIGHT",    # TURN_RIGHT
    3: "PALM_LEFT",     # TURN_LEFT
    4: "ONE",           # SPEED_UP
    5: "TWO",           # SLOW_DOWN
}

# Try loading SVM model
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"[Client] Loaded gesture classification model: {MODEL_PATH}")
except Exception as e:
    print(f"[Client] Failed to load model {MODEL_PATH}, fallback to rule-based recognition: {e}")
    ml_model = None

HOST = '127.0.0.1'
PORT = 10020

# ========= Feature Extraction (consistent with training) =========
def extract_hand_features(landmarks):
    """
    landmarks: hand_landmarks.landmark (length 21)
    Features: 21 keypoints (x,y), normalized relative to the wrist → 42 dimensions.
    """
    wrist = landmarks[0]
    wx, wy = wrist.x, wrist.y

    # Measure of hand size (wrist → middle finger tip)
    middle_tip = landmarks[12]
    hand_size = np.sqrt((middle_tip.x - wx) ** 2 + (middle_tip.y - wy) ** 2)
    if hand_size < 1e-6:
        hand_size = 1e-6  # avoid division by zero

    features = []
    for p in landmarks:
        rx = (p.x - wx) / hand_size   # translation + scale normalization
        ry = (p.y - wy) / hand_size
        features.extend([rx, ry])

    return np.array(features, dtype=np.float32)

# ========= Rule-Based Gesture Recognition =========
def recognize_gesture_rule_based(landmarks, hand_label):
    """
    Basic rule-based recognition using finger openness.
    """
    finger_tips = [4, 8, 12, 16, 20]
    finger_dips = [2, 6, 10, 14, 18]
    finger_states = []

    # Thumb
    thumb_tip = landmarks[finger_tips[0]]
    thumb_dip = landmarks[finger_dips[0]]
    if hand_label == "Right":
        finger_states.append(1 if thumb_tip.x > thumb_dip.x else 0)
    else:
        finger_states.append(1 if thumb_tip.x < thumb_dip.x else 0)

    # Other fingers
    for i in range(1, 5):
        tip = landmarks[finger_tips[i]]
        dip = landmarks[finger_dips[i]]
        finger_states.append(1 if tip.y < dip.y else 0)

    count = sum(finger_states)

    if count == 0:
        return "FIST"
    elif count == 5:
        return "PALM"
    elif finger_states[1] == 1 and count == 1:
        return "ONE"
    elif finger_states[1] == 1 and finger_states[2] == 1 and count == 2:
        return "TWO"
    elif count == 3:
        return "THREE"
    elif count == 4:
        return "FOUR"
    else:
        return "UNKNOWN"

# ========= SVM-Based Gesture Recognition =========
def recognize_gesture_ml(landmarks):
    """
    Use SVM model to classify gesture.
    Returns gesture label string or None if classification fails.
    """
    if ml_model is None:
        return None

    features = extract_hand_features(landmarks)
    try:
        label = int(ml_model.predict([features])[0])  # 0~5
        return ML_LABELS.get(label, None)
    except Exception as e:
        print(f"[Client] SVM prediction failed. Fallback to rule-based: {e}")
        return None

# ========= Command Mapping =========
def map_gesture_to_command(gesture: str) -> str:
    mapping = {
        "FIST": "STOP",
        "PALM_FORWARD": "FORWARD",
        "PALM_RIGHT": "TURN_RIGHT",
        "PALM_LEFT": "TURN_LEFT",
        "ONE": "SPEED_UP",
        "TWO": "SLOW_DOWN",
    }
    return mapping.get(gesture, "")

# ========= Main Program =========
def main():
    # Connect to robot controller server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"[Client] Connected to {HOST}:{PORT}")

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[Client] Failed to open camera")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    stable_gesture = None
    gesture_buffer = {"gesture": None, "count": 0}
    STABLE_THRESHOLD = 3
    last_command = None

    print("[Client] Camera activated, press 'q' to exit")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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

                current_frame_gesture = None

                # ===== Hand detection =====
                if results.multi_hand_landmarks and results.multi_handedness:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_label = results.multi_handedness[0].classification[0].label
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # 1. Try ML prediction first
                    g_ml = recognize_gesture_ml(hand_landmarks.landmark)
                    if g_ml is not None:
                        current_frame_gesture = g_ml
                    else:
                        # 2. Fallback to rule-based method
                        current_frame_gesture = recognize_gesture_rule_based(
                            hand_landmarks.landmark, hand_label
                        )

                # ===== Stable gesture logic =====
                if current_frame_gesture is not None:
                    if stable_gesture is None:
                        stable_gesture = current_frame_gesture
                        print(f"[Client] Initial stable gesture: {stable_gesture}")
                    elif current_frame_gesture != stable_gesture:
                        if current_frame_gesture == gesture_buffer["gesture"]:
                            gesture_buffer["count"] += 1
                        else:
                            gesture_buffer["gesture"] = current_frame_gesture
                            gesture_buffer["count"] = 1

                        if gesture_buffer["count"] >= STABLE_THRESHOLD:
                            stable_gesture = gesture_buffer["gesture"]
                            gesture_buffer = {"gesture": None, "count": 0}
                            print(f"[Client] Stable gesture switched to: {stable_gesture}")

                            cmd = map_gesture_to_command(stable_gesture)
                            if cmd and cmd != last_command:
                                sock.sendall(cmd.encode('utf-8'))
                                print(f"[Client] Sent command: {cmd}")
                                last_command = cmd
                    else:
                        gesture_buffer = {"gesture": None, "count": 0}

                # ===== Display status =====
                cv2.putText(
                    frame,
                    f"Stable: {stable_gesture if stable_gesture else 'None'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Gesture Client", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            sock.close()
            print("[Client] Exited")

if __name__ == "__main__":
    main()
