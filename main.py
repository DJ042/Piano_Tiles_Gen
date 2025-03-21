# Final Step: Run the Full Pipeline
# To automate everything, you can create a main script.

import os
from src.preprocess_audio import audio_to_spectrogram
from src.preprocess_midi import midi_to_tile_sequence
from src.train_model import model
from src.predict_tiles import model as predict_model

# Step 1: Convert Audio to Spectrogram
print("Processing Audio...")
audio_to_spectrogram("data/example.wav")

# Step 2: Convert MIDI to Tile Sequence
print("Processing MIDI...")
midi_to_tile_sequence("data/example.mid")

# Step 3: Train the Model
print("Training Model...")
model.fit(...)  # Call model training

# Step 4: Predict Tiles for New Audio
print("Predicting Tiles...")
predicted_tiles = predict_model.predict(...)
print(predicted_tiles)
