import os
import librosa

data_dir = "Audio"
classes = os.listdir(data_dir)

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(class_dir, filename)
            audio_data, sample_rate = librosa.load(file_path, sr=16000)

            if len(audio_data)/sample_rate < 5:
                os.remove(file_path)