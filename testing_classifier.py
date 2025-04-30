import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dictionary = pickle.load(open('./model.p', 'rb'))
model = model_dictionary['model']

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands # Initializes the hand detection solution from mediapipe
mp_drawing = mp.solutions.drawing_utils # Provides tools to draw th detected landmarks and their connections on images
mp_drawing_styles = mp.solutions.drawing_styles # Provides predefined drawing styles for landmark visualization.

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # 'static_image_mode' indicates that the input is a static image (not video). 'min_detection_confidence' sets the confidence threshold for detecting hands. Only hands detected with a confidence higher than 30% will be proceesed.

labels_dictionary = {
0: 'A',
1: 'B',
2: 'C',
3: 'D',
4: 'E',
5: 'F',
6: 'G',
7: 'H',
8: 'I',
9: 'J',
10: 'K',
11: 'L',
12: 'M',
13: 'N',
14: 'O',
15: 'P',
16: 'Q',
17: 'R',
18: 'S',
19: 'T',
20: 'U',
21: 'V',
22: 'W',
23: 'X',
24: 'Y',
25: 'Z'
}

while True:

    data_aux = []
    x_ = []
    y_ = []

    success, frame = capture.read()

    H, W, _ = frame.shape

    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_RGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        for hand_landmarks in results.multi_hand_landmarks: 
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        data_aux.append(x) 
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

        padding = 20
        x1 = max(0, int(min(x_) * W) - padding)
        y1 = max(0, int(min(y_) * H) - padding)
        x2 = min(W, int(max(x_) * W) + padding)
        y2 = min(H, int(max(y_) * H) + padding)

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dictionary[int(prediction[0])]

        # Frame around the hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()