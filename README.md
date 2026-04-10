# Naruto Hand Jutsu Recognition

A machine learning project that recognizes Naruto hand signs (jutsus) in real-time using computer vision and MediaPipe hand tracking.

## Features

- Real-time hand gesture recognition using webcam
- Trained on 12 Naruto hand signs: Bird, Boar, Dog, Dragon, Hare, Horse, Monkey, Ox, Ram, Rat, Snake, Tiger
- Uses MediaPipe for hand landmark detection
- MLPClassifier for gesture classification
- Stability buffer to prevent flickering predictions

## Installation

1. Clone or download this repository.
2. Install Python dependencies:
   ```
   pip install -r req.txt
   ```

## Dataset

The dataset consists of images of hand gestures corresponding to Naruto jutsus. Each subfolder in `dataset/` contains images for one gesture:

- bird/
- boar/
- dog/
- dragon/
- hare/
- horse/
- monkey/
- ox/
- ram/
- rat/
- snake/
- tiger/

## Training

To train the model:

1. Ensure your dataset is in the `dataset/` folder.
2. Run the training script:
   ```
   python train.py
   ```
   This will process the images, extract hand landmarks, train the model, and save `model.pkl` and `scaler.pkl`.

## Prediction

To run real-time prediction:

1. Ensure `model.pkl` and `scaler.pkl` are present (from training).
2. Run the prediction script:
   ```
   python predict.py
   ```
   This will open your webcam and display the recognized jutsu on screen. Press 'q' to quit.

## Requirements

- Python 3.x
- Webcam for real-time prediction
- Dependencies: OpenCV, MediaPipe, NumPy, Scikit-learn

## Notes

- The model is trained to recognize single or dual-hand gestures.
- Predictions are stabilized to reduce noise.
- For best results, ensure good lighting and clear hand visibility.
