import cv2
import mediapipe as mp
import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

DATASET_PATH = "dataset"

data = []
labels = []

print("📸 Reading dataset...")

for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hands_sorted = sorted(
                results.multi_hand_landmarks,
                key=lambda h: h.landmark[0].x
            )

            landmarks = []

            for hand_landmarks in hands_sorted:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # 🔥 FIX: ensure 2 hands
            if len(hands_sorted) == 1:
                landmarks.extend([0] * 63)

            # normalize
            base_x = landmarks[0]
            base_y = landmarks[1]

            for i in range(0, len(landmarks), 3):
                landmarks[i] -= base_x
                landmarks[i+1] -= base_y

            # scale normalize
            max_value = max(abs(x) for x in landmarks)
            if max_value != 0:
                landmarks = [x / max_value for x in landmarks]

            data.append(landmarks)
            labels.append(label)

print("✅ Total samples:", len(data))

X = np.array(data)
y = np.array(labels)

# scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

from collections import Counter
print("Class distribution:", Counter(labels))

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# train
print("🤖 Training model...")


model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500
)

model.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print("🎯 Accuracy:", model.score(X_test, y_test))
print("\n📊 Detailed Report:\n")
print(classification_report(y_test, y_pred))

# save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("💾 Model + scaler saved")