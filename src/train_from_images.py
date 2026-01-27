import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

DATA_DIR = '../data/images'  # Where you pasted the folder

data = []
labels = []

print("Loading images and extracting landmarks... This may take a while.")

# 2. Loop through every letter folder (A, B, C...)
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Directory not found: {DATA_DIR}")
    print("Did you paste the 'images' folder into 'data'?")
    exit()

for dir_ in os.listdir(DATA_DIR):
    # Skip hidden files
    if dir_.startswith('.'): continue
    
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path): continue

    print(f"Processing Class: {dir_}")
    
    # Loop through every image in that letter's folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Convert to RGB (MediaPipe requires RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract Landmarks
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append(lm.x)
                    lm_list.append(lm.y)
                
                data.append(lm_list)
                labels.append(dir_) # Use folder name (e.g., 'A') as label

# 3. Train the Model
data = np.asarray(data)
labels = np.asarray(labels)

print(f"\nData loading complete.")
print(f"Total Samples: {len(data)}")

if len(data) > 0:
    print("Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100) # 100 trees
    model.fit(data, labels)

    # Save the model
    if not os.path.exists('../models'):
        os.makedirs('../models')

    with open('../models/isl_model.p', 'wb') as f:
        pickle.dump(model, f)
        
    print("SUCCESS: Model trained on new data and saved to models/isl_model.p")
else:
    print("ERROR: No landmarks found. Check if your images are valid.")