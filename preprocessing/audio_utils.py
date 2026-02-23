import torch
import torchaudio

SAMPLE_RATE = 16000
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def load_wav(path):
    waveform, sr = torchaudio.load(path)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0)  # mono

    if waveform.shape[0] < SAMPLES:
        pad = torch.zeros(SAMPLES - waveform.shape[0])
        waveform = torch.cat([waveform, pad])
    else:
        waveform = waveform[:SAMPLES]

    return waveform