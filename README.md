# Piano_Tiles_Gen
Machine Learning to make a Piano Tiles Generation software


PIANO_TILES_ML/ \n
│── data/
│   ├── example.mid          # Sample MIDI file
│   ├── example.wav          # Sample audio file
│   ├── spectrograms.npy     # Preprocessed spectrograms
│   ├── tile_sequences.npy   # Preprocessed tile sequences
│── src/
│   ├── preprocess_audio.py  # Convert audio to spectrogram
│   ├── preprocess_midi.py   # Convert MIDI to tile sequences
│   ├── train_model.py       # CNN+LSTM training
│   ├── predict_tiles.py     # Predict tiles from new audio
│── main.py                  # Run the full pipeline
│── requirements.txt         # Required Python libraries
│── README.md                # Project description
