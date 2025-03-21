# Step 3: Train the Model
# Uses spectrograms and tile sequences to train a CNN + LSTM model.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten

# Load data
X = np.load("../data/spectrograms.npy")
y = np.load("../data/tile_sequences.npy")

# Reshape for training
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    Flatten(),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(4, activation='softmax')  # 4 tile positions
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save("../data/piano_tiles_model.h5")
