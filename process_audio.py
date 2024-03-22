import os
import librosa
import numpy as np
from tensorflow.image import resize
from tensorflow.keras.utils import pad_sequences

def process_audio(audio_data, sample_rate, target_shape=(128, 128)):
    length = len(audio_data)/sample_rate
    max_length = 5*sample_rate
    if length > 5:
        audio_data = audio_data[:max_length]

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return mel_spectrogram

def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=16000)
                data.append(process_audio(audio_data, sample_rate, target_shape))
                labels.append(i)

    return np.array(data), np.array(labels)

def load_and_process_single(file_path, target_shape=(128, 128)):
    audio_data, sample_rate = librosa.load(file_path, sr=16000)
    processed_audio = process_audio(audio_data, sample_rate, target_shape)
    return np.array([processed_audio])