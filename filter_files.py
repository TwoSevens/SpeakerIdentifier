import os
import librosa
from process_audio import get_mean_rms

data_dir = "Audio"
classes = sorted(os.listdir(data_dir))
threshold = 1E-3

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(class_dir, filename)
            audio_data, sample_rate = librosa.load(file_path, sr=16000)

            if get_mean_rms(audio_data) < threshold:
                print("Removed", file_path)
                os.remove(file_path)