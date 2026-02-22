#feature_extraction.py
import librosa
import numpy as np
from preprocessing.audio_utils import load_audio

def extract_features(path):
    y, sr = load_audio(path)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        n_fft=512,
        hop_length=256
    )

    return np.mean(mfcc.T, axis=0)