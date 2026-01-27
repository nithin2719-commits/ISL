import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands  # Enabled 2 hands here
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0):
        # We only return the landmarks for ONE hand to keep the model happy
        lm_list = []
        if self.results.multi_hand_landmarks:
            # Always grab the first hand detected (the 'dominant' hand)
            my_hand = self.results.multi_hand_landmarks[0]
            
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append(lm.x)
                lm_list.append(lm.y)
                
        return lm_list