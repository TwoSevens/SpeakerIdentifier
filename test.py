import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from process_audio import *
import numpy as np

names = ["Alasfoor", "Ali Ayyad", "Ali Jaffar", "Elyas"]
model_path = os.path.join("Models", sorted(os.listdir("Models"))[-1])
model = load_model(model_path)
print("loaded", model_path)

while True:
    audio_path = input("Enter path: ")
    aud = load_and_process_single(audio_path, (128, 128))
    prediction = model.predict([aud], verbose=0)[0]

    for index, certainty in enumerate(prediction):
        print(names[index], round(certainty*100, 2))

    print()
    

