# Step 2: Convert MIDI to Tile Sequences
# Extracts note timings and maps them to tile positions.

import mido
import numpy as np
import pandas as pd

def midi_to_tile_sequence(midi_file):
    mid = mido.MidiFile(midi_file)
    tiles = []
    
    current_time = 0
    for msg in mid.tracks[1]:  
        if msg.type == 'note_on':  
            current_time += msg.time  
            tiles.append((msg.note, current_time))

    df = pd.DataFrame(tiles, columns=['Note', 'Time'])
    df['Tile_Position'] = df['Note'] % 4  # Assign to 4 tile lanes
    return df

# Save tile sequences
midi_file = "../data/example.mid"
df_tiles = midi_to_tile_sequence(midi_file)
np.save("../data/tile_sequences.npy", df_tiles.to_numpy())
