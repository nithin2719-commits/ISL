import cv2
import os
from capture import HandTracker  # Import your tracker

# Define where to save images
DATA_DIR = '../data/images'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize Hand Tracker
tracker = HandTracker()
cap = cv2.VideoCapture(0)

def collect_for_class(class_name):
    # Create the folder for this specific letter
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f"\n--- COLLECTING: {class_name} ---")
    print("Press 'q' to start capturing.")

    # 1. PREVIEW PHASE (Get your hand ready)
    while True:
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        
        # Draw landmarks so you know it sees you
        tracker.find_hands(frame)
        
        cv2.putText(frame, f"Ready? Press 'q' for: {class_name}", (40, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 2. CAPTURE PHASE (Save 100 images)
    count = 0
    while count < 100:
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        
        # We save the CLEAN image (without lines) for training accuracy
        save_frame = frame.copy()
        
        # But we show you the lines so you can aim
        tracker.find_hands(frame)
        
        # Save to folder
        img_