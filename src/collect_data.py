import cv2
import os
import pickle
import numpy as np
from capture import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    data_dir = '../data'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("ISL Data Collection")
    print("-------------------")
    
    while True:
        name = input("Enter the name of the sign (e.g., 'A', 'Hello') or 'q' to quit: ").strip()
        if name.lower() == 'q':
            break
        if not name:
            continue
            
        dataset_size = 100
        data = []
        
        print(f"\nCollecting data for '{name}'.")
        print("1. Position your hand.")
        print("2. Press 's' when ready to start recording.")
        print("3. Press 'q' to cancel this sign.")

        # Wait for start
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame = tracker.find_hands(frame)
            
            cv2.putText(frame, f"Target: {name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "Press 's' to start", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Data Collection", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                break
            if key & 0xFF == ord('q'):
                data = None # Signal to skip
                break
        
        if data is None:
            print("Cancelled.")
            continue

        # Collect data
        counter = 0
        while counter < dataset_size:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame = tracker.find_hands(frame)
            lm_list = tracker.find_position(frame)
            
            if len(lm_list) != 0:
                data.append(lm_list)
                counter += 1
            
            cv2.putText(frame, f"Collecting {name}: {counter}/{dataset_size}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)
            
        # Save
        file_path = os.path.join(data_dir, f'{name}.pickle')
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {dataset_size} samples for '{name}' to {file_path}\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()