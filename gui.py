import os
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import librosa
from tensorflow import keras
from tensorflow.keras.models import load_model
from process_audio import *
import numpy as np

class AttendanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Attendance Taker")

        self.speakers = sorted(os.listdir("Audio"))

        self.checkboxes = []
        for speaker in self.speakers:
            var = tk.IntVar()
            checkbox = tk.Checkbutton(master, text=speaker, variable=var)
            checkbox.pack(anchor=tk.W)
            self.checkboxes.append((speaker, var))

        self.record_button = tk.Button(master, text="Record Audio", command=self.record_audio)
        self.record_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.model_path = os.path.join("Models", sorted(os.listdir("Models"))[-1])
        self.model = load_model(self.model_path)
        print("Loaded", self.model_path)

    def record_audio(self):
        filename = "recorded_audio.wav"
        audio, sr = record_audio(filename)

        aud = process_audio(audio, sr, (128, 128))
        prediction = self.model.predict(np.array([aud]), verbose=0)[0]

        for pre in prediction:
            print(pre*100)

        most_likely_speaker_index = np.argmax(prediction)
        most_likely_speaker = self.speakers[most_likely_speaker_index]

        for speaker, var in self.checkboxes:
            if speaker == most_likely_speaker:
                var.set(1)
                break
            else:
                pass

    def reset(self):
        for speaker, var in self.checkboxes:
            var.set(0)

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


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()