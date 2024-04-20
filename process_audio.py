import os
import sys
import librosa
import numpy as np
from tensorflow.image import resize
from tensorflow.keras.utils import pad_sequences
import random

def get_mean_rms(audio_data):
    rms = librosa.feature.rms(y=audio_data)
    return rms.mean()

def process_audio(audio_data, sample_rate, target_shape=(128, 128)):
    length = len(audio_data)/sample_rate
    max_length = 5*sample_rate
    if length > 5:
        audio_data = audio_data[:max_length]

    if get_mean_rms(audio_data) < 1E-2:
        audio_data = audio_data * 2

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return mel_spectrogram

def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128), samples_per_person=sys.maxsize):
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        samples = os.listdir(class_dir)
        samples = samples[:min(len(os.listdir(class_dir)), samples_per_person)]
        random.shuffle(samples)
        for filename in samples:
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