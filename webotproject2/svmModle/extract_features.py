import numpy as np

def extract_hand_features(landmarks):
    """
    landmarks: hand_landmarks.landmark (length = 21)
    Feature extraction method:
    Relative coordinates based on the wrist (landmark 0) as the origin,
    normalized by an estimated hand size, producing a 42-dimensional feature vector.
    """
    wrist = landmarks[0]
    wx, wy = wrist.x, wrist.y

    # Compute hand scale: distance from wrist to middle fingertip
    middle_tip = landmarks[12]
    hand_size = np.sqrt((middle_tip.x - wx) ** 2 + (middle_tip.y - wy) ** 2)
    if hand_size < 1e-6:
        hand_size = 1e-6  # Prevent division by zero

    features = []
    for p in landmarks:
        rx = (p.x - wx) / hand_size   # translation + scale normalization
        ry = (p.y - wy) / hand_size
        features.extend([rx, ry])

    return np.array(features, dtype=np.float32)
