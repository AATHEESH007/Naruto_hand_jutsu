import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# 🔥 Stability buffer (prevents flicker)
history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hands_sorted = sorted(
        results.multi_hand_landmarks,
        key=lambda h: h.landmark[0].x
        )[:2]

        landmarks = []

        for hand_landmarks in hands_sorted:

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) != 126:
                continue


        if len(hands_sorted) == 1:
            landmarks.extend([0] * 63)


        base_x = landmarks[0]
        base_y = landmarks[1]

        for i in range(0, len(landmarks), 3):
            landmarks[i] -= base_x
            landmarks[i+1] -= base_y


        max_value = max(abs(x) for x in landmarks)
        if max_value != 0:
            landmarks = [x / max_value for x in landmarks]


        landmarks = scaler.transform([landmarks])


        probs = model.predict_proba(landmarks)[0]
        max_prob = np.max(probs)
        pred = model.classes_[np.argmax(probs)]


        if max_prob > 0.6:
            history.append(pred)

            # 🔥 Stability check
            if history.count(pred) > 5:
                cv2.putText(
                    frame,
                    f"{pred} ({max_prob:.2f})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
    else:
        history.clear()

    cv2.imshow("Naruto Jutsu Detection (2 Hands)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()