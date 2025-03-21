# Step 4: Predict Tiles for New Audio
# Loads the trained model and predicts tiles for a new song.

import numpy as np
import tensorflow as tf
from preprocess_audio import audio_to_spectrogram

# Load trained model
model = tf.keras.models.load_model("../data/piano_tiles_model.h5")

# Convert new audio to spectrogram
audio_file = "../data/new_song.wav"
spectrogram = audio_to_spectrogram(audio_file)

# Reshape and predict
spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)
predicted_tiles = model.predict(spectrogram)
predicted_tiles = np.argmax(predicted_tiles, axis=1)

# Output predictions
print(predicted_tiles)
