import torch
import torchaudio

def extract_mfcc(waveform, sample_rate=16000, n_mfcc=40):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 512, "hop_length": 256, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform)
    return mfcc.squeeze(0).transpose(0, 1)  # shape: (time, n_mfcc)