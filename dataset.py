import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

data_directory = './data'

# Lists to store landmark data and corresponding labels
data = []
labels = []

# Traverse each folder in the data directory
for directory in os.listdir(data_directory):
    dir_path = os.path.join(data_directory, directory)

    # Skip if it's not a directory or if it's a hidden directory (like .DS_Store)
    if not os.path.isdir(dir_path) or directory.startswith('.'):
        continue

    for image_path in os.listdir(dir_path):
        image_file_path = os.path.join(dir_path, image_path)
        image = cv2.imread(image_file_path)
        if image is None:
            print(f"Warning: Could not read image {image_file_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

                data.append(data_aux)
                labels.append(directory)

# Save the collected data into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset creation complete. Total samples: {len(data)}")
