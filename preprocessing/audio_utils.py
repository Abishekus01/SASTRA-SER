import os
import torch
import torchaudio

def load_wav(file_path, target_sample_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)(waveform)
    # Make mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform