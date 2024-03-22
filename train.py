import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from process_audio import *
from datetime import datetime
import random

audio_data_shape = (128, 128)
classes = ["Alasfoor", "Ali Ayyad", "Ali Jaffar", "Elyas"]
data, labels = load_and_preprocess_data("Audio", classes, target_shape=audio_data_shape)
labels = to_categorical(labels, num_classes=len(classes))
print(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=random.seed())

model = Sequential([
    Input(shape=X_train[0].shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(random.random()*0.2, seed=random.seed()),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))
model.save("Models/{:%Y.%m.%d__%H_%M}.keras".format(datetime.now()))