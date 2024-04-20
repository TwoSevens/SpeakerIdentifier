import os
import math
import librosa
import soundfile as sf

path = "NoTrain/Yaseen"
new_path = "Audio/Yaseen"
clip_length = 5

global_counter = 1
for audio_file in os.listdir(path):
    file_path = os.path.join(path, audio_file)

    audio_data, sr = librosa.load(file_path)
    length = len(audio_data)/sr

    for i in range(0, math.ceil(length), clip_length):
        clip_path = os.path.join(new_path, f"{audio_file[:-4]}_{global_counter}.wav")
        clip = audio_data[i*sr:(i+clip_length)*sr]

        sf.write(clip_path, clip, sr, format='wav')
        global_counter += 1
