import torch
import torchaudio
import torchaudio.transforms as T

def extract_log_mel(waveform):
    # Ensure waveform is 2D (Channel, Time) for torchaudio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # 1. Generate Mel Spectrogram (128 bins to match SwinTSER paper)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )(waveform)

    # 2. Convert to Decibels (Log scale) - CRITICAL for training
    # Added top_db for cleaner contrast
    log_mel = T.AmplitudeToDB(top_db=80)(mel_spectrogram)

    # 3. Shape for Model: (Time, Mel)
    log_mel = log_mel.squeeze(0).transpose(0, 1)

    return log_mel