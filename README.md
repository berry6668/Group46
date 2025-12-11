# Group46 – GestureCommand Project
## Overview

GestureCommand is the final project for the Intelligent Robotics module at the University of Birmingham.
The system enables real-time hand-gesture control of a differential-drive robot inside a Webots simulation environment.

Using MediaPipe Hands, the system extracts 21 hand landmarks and classifies six static gestures with an SVM (RBF kernel). These gestures are translated into robot commands and transmitted via socket communication to a Webots controller, enabling intuitive and natural interaction.

The project also includes controlled experiments and data analysis to compare gesture control vs. keyboard control, as well as the impact of LED feedback on user confidence and error rates.

## Key Features
### 1. Hand Gesture Recognition

Uses MediaPipe Hands for landmark extraction

Custom feature vector (distances, relative coordinates, angles)

SVM classifier trained to recognize six gestures:

**Fist** – Stop

**Palm Forward** – Move forward

**Palm Left** – Turn left

**Palm Right** – Turn right

**One Finger** – Speed up

**Two Fingers** – Slow down

### 2. Control Architecture

Complete **Gesture → Command → Robot** real-time pipeline

Socket communication between client and Webots controller

Differential-drive control

LED feedback indicating recognition status

### 3. Experimentation & Analysis

Performance comparison:

Gesture control vs. keyboard control

LED vs. non-LED feedback conditions

Metrics include:

Task completion time

Collision count

Parking success rate

Confusion matrix & classification scores

# Repository Structure
```
Group46/
└── webotproject2/
    ├── gesture_client.py                    # Front-end gesture recognition & command sender
    │
    ├── controllers/gesture_cam/             # Webots robot controller & experiment analysis
    │   ├── gesture_cam.py                   # Receives commands, controls robot & LED feedback
    │   └── analyse_time.py                  # Computes experiment time statistics
    │
    ├── svmModle/                            # SVM training and evaluation scripts (note spelling)
    │   ├── train_svm.py                     # Trains the SVM gesture classifier
    │   ├── collect_svm_data.py              # Collects training dataset
    │   └── TEST/                            # Testing and evaluation utilities
    │       ├── svm test.py                  # Tests classification performance
    │       └── svm_confusion_matrix.py      # Generates confusion matrix & plots
    │
    ├── lib/                                 # C-based robot control library
    │   ├── Makefile                         # Build configuration
    │   ├── backprop.h                       # Header file
    │   └── odometry_goto.c                  # Odometry & motion control logic
    │
    ├── worlds/
    │   └── gesture_world.wbt                # Webots simulation world
    │
    ├── protos/
    │   └── E-puck.proto                     # E-puck robot model
    │
    └── ...                                  # Additional logs, artifacts, etc.
```
## Usage Instructions
### 1. Environment Setup

Install Python 3.9.13 and required libraries:

`pip install opencv-python mediapipe numpy pandas matplotlib scikit-learn joblib`

### 2. Running the Simulation

1.Open Webots

2.Load the world file:
```
worlds/gesture_world.wbt
```

3.Set the controller:
```
controllers/gesture_cam/gesture_cam.py
```

4.Click Run (Play)

### 3. Starting the Gesture Client

Run:
```
python gesture_client.py
```
This activates the webcam, detects gestures, classifies them, and sends commands to Webots.

## Model Training & Testing
### Train SVM model
```
python svmModle/train_svm.py
```
### Collect training samples
```
python svmModle/collect_svm_data.py
```
### Evaluate / generate confusion matrix
```
python svmModle/TEST/svm test.py
python svmModle/TEST/svm_confusion_matrix.py
```
## Experiment Analysis

To compute and visualize performance metrics:
```
python controllers/gesture_cam/analyse_time.py
```
This script compares keyboard vs. gesture control and evaluates LED feedback.

## Acknowledgements
This project was developed by Group 46.

Group member:Yiyan Ge, Ge Lin, Zhu Jiang, Tianle Feng

We thank the Intelligent Robotics teaching team for their guidance and support.
