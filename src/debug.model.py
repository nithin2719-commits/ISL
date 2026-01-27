import pickle
import numpy as np
import os

model_path = '../models/isl_model.p'

print(f"Checking model at: {model_path}")

if not os.path.exists(model_path):
    print("ERROR: Model file does not exist!")
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model Loaded Successfully!")
    
    # Check what classes the model knows
    if hasattr(model, 'classes_'):
        print(f"Classes known by model: {model.classes_}")
        print(f"Total classes: {len(model.classes_)}")
    else:
        print("This model does not have class names stored.")

    # Check Data Folder
    data_path = '../data/images'
    if os.path.exists(data_path):
        print(f"\nChecking Data Folder: {data_path}")
        folders = os.listdir(data_path)
        print(f"Found folders: {folders}")
        
        # Check count of first folder
        if len(folders) > 0:
            first_folder = os.path.join(data_path, folders[0])
            if os.path.isdir(first_folder):
                count = len(os.listdir(first_folder))
                print(f"Folder '{folders[0]}' has {count} images.")
    else:
        print("\nWARNING: Data folder not found at ../data/images")