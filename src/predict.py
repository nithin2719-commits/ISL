import pickle
import numpy as np
import os

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'isl_model.p')

class SignClassifier:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully.")
        else:
            print("Warning: Model file not found.")

    def predict(self, landmarks):
        if self.model is None or len(landmarks) == 0:
            return ""

        input_data = np.array([landmarks])
        
        try:
            # The model now returns the label string directly (e.g., 'A')
            prediction = self.model.predict(input_data)
            return prediction[0]
        except Exception as e:
            return ""