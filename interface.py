import os
import sounddevice as sd
import soundfile as sf
import librosa
from tensorflow import keras
from tensorflow.keras.models import load_model
from process_audio import *
import numpy as np

def record_audio(filename, duration=5, samplerate=16000, channels=2):
    print("Recording audio for {} seconds...".format(duration))
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, blocking=True)
    print("Recording finished.")

    audio_data *= 3
    
    # Save the recorded audio to a file
    sf.write(filename, audio_data, samplerate)

    # Load the saved audio file using librosa
    audio, sr = librosa.load(filename, sr=samplerate)
    return audio, sr

names = os.listdir('Audio')
model_path = os.path.join("Models", sorted(os.listdir("Models"))[-2])
latest_model = load_model(model_path)
best_model = load_model( os.path.join("Models", "checkpoint.keras"))
print("loaded", model_path)

while True:
    input("Enter when ready")
    filename = "recorded_audio.wav"
    audio, sr = record_audio(filename)

    aud = process_audio(audio, sr, (128, 128))
    aud = np.array([aud])
    best_prediction = best_model.predict(aud, verbose=0)[0]
    prediction = latest_model.predict(aud, verbose=0)[0]

    print("Latest Model".center(20, "-"))
    for index, certainty in enumerate(prediction):
        print(names[index], round(certainty*100, 2))

    print("Best Model".center(20, "-"))
    for index, certainty in enumerate(best_prediction):
        print(names[index], round(certainty*100, 2))

    print()