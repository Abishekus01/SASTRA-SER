#audio_utils.py
import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3   # seconds
SAMPLES = SAMPLE_RATE * DURATION

def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < SAMPLES:
        pad_width = SAMPLES - len(y)
        y = np.pad(y, (0, pad_width))
    else:
        y = y[:SAMPLES]

    return y, sr