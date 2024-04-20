import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from process_audio import *
import numpy as np

names = os.listdir('Audio')
model_path = os.path.join("Models", sorted(os.listdir("Models"))[-2])
latest_model = load_model(model_path)
best_model = load_model( os.path.join("Models", "checkpoint.keras"))
print("loaded", model_path)

while True:
    audio_path = input("Enter path: ")
    aud = load_and_process_single(audio_path, (128, 128))
    best_prediction = best_model.predict([aud], verbose=0)[0]
    prediction = latest_model.predict([aud], verbose=0)[0]

    print("Latest Model".center(20, "-"))
    for index, certainty in enumerate(prediction):
        print(names[index], round(certainty*100, 2))

    print("Best Model".center(20, "-"))
    for index, certainty in enumerate(best_prediction):
        print(names[index], round(certainty*100, 2))

    print()