import pyttsx3

try:
    print("Initializing Voice Engine...")
    engine = pyttsx3.init()
    
    # Force the Windows 'SAPI5' driver (Standard for Windows)
    # If this fails, we will try without it
    
    print("Testing Voice...")
    engine.say("Testing audio. Can you hear me?")
    engine.runAndWait()
    
    print("Success! You should have heard that.")

except Exception as e:
    print("\n--- ERROR FOUND ---")
    print(e)
    print("-------------------")