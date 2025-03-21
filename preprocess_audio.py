# Step 1: Convert Audio to Spectrogram
# Run this first to extract features from audio.

import librosa
import numpy as np

def audio_to_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Save spectrograms
audio_file = "../data/example.wav"
spectrogram = audio_to_spectrogram(audio_file)
np.save("../data/spectrograms.npy", spectrogram)
