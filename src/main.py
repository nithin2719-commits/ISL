import cv2
import pyttsx3
import time
from capture import HandTracker
from predict import SignClassifier

def main():
    # 1. Setup Camera and AI
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=2) # Enable 2 hands (visual only)
    classifier = SignClassifier()
    
    # 2. Setup Voice Engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) # Speed of speech
    
    # --- VARIABLES ---
    current_sentence = ""
    last_prediction = ""
    frame_count = 0
    FRAME_THRESHOLD = 15  # Adjust this: Higher = Slower but more accurate
    
    print("--- CONTROLS ---")
    print("Spacebar : Add Space")
    print("Backspace: Delete Letter")
    print("Enter    : Speak Full Sentence")
    print("'c'      : Clear All")
    print("'q'      : Quit")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Detect Hands
        frame = tracker.find_hands(frame)
        lm_list = tracker.find_position(frame)
        
        raw_pred = ""
        
        # Make Prediction if hand is seen
        if len(lm_list) > 0:
            raw_pred = classifier.predict(lm_list)
        
        # --- STABILITY LOGIC ---
        if raw_pred == last_prediction and raw_pred != "":
            frame_count += 1
        else:
            frame_count = 0
            last_prediction = raw_pred
            
        # If sign is held steady...
        if frame_count == FRAME_THRESHOLD:
            stable_letter = raw_pred
            
            # Logic to avoid repeating the same letter instantly (like AAAAA)
            if len(current_sentence) == 0 or current_sentence[-1] != stable_letter:
                current_sentence += stable_letter
                
                # SPEAK THE LETTER (Immediate Feedback)
                engine.say(stable_letter)
                engine.runAndWait()

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'): # Quit
            break
        elif key & 0xFF == 32: # SPACEBAR (ASCII 32)
            current_sentence += " "
        elif key & 0xFF == 8:  # BACKSPACE (ASCII 8)
            current_sentence = current_sentence[:-1]
        elif key & 0xFF == ord('c'): # 'c' to Clear
            current_sentence = ""
        elif key & 0xFF == 13: # ENTER KEY (ASCII 13)
            # Speak the FULL sentence
            print(f"Speaking: {current_sentence}")
            engine.say(current_sentence)
            engine.runAndWait()

        # --- DISPLAY UI ---
        # Top Bar (Debug)
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), cv2.FILLED)
        bar_width = int((frame_count / FRAME_THRESHOLD) * 200)
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f"Detecting: {last_prediction}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Bottom Bar (Sentence)
        cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"Sentence: {current_sentence}", (20, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ISL Translator Final", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()