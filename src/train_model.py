import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from capture import HandTracker

# Define the gestures to train
CLASSES = ['A', 'B', 'C', 'Hello', 'Yes'] 
DATA_SAMPLES = 100 

def collect_data():
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)
    data = []
    labels = []

    for label_name in CLASSES:
        print(f"Collecting data for: {label_name}. Press 'q' to start.")
        while True:
            ret, frame = cap.read()
            if not ret: continue
            cv2.putText(frame, f"Press 'q' to collect: {label_name}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count = 0
        while count < DATA_SAMPLES:
            ret, frame = cap.read()
            if not ret: continue
            tracker.find_hands(frame)
            lm_list = tracker.find_position(frame)

            if len(lm_list) > 0:
                data.append(lm_list)
                labels.append(label_name) 
                count += 1
                cv2.putText(frame, f"Count: {count}/{DATA_SAMPLES}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)
            
    cap.release()
    cv2.destroyAllWindows()
    return np.array(data), np.array(labels)

def train_model(data, labels):
    print("Training Model...")
    model = RandomForestClassifier()
    model.fit(data, labels)
    
    if not os.path.exists('../models'):
        os.makedirs('../models')
    
    # Save model using pickle
    with open('../models/isl_model.p', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved to models/isl_model.p")

if __name__ == "__main__":
    X, y = collect_data()
    train_model(X, y)