import torch
import torchaudio

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=128
)

amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def extract_log_mel(waveform):
    mel = mel_transform(waveform)
    log_mel = amplitude_to_db(mel)
    log_mel = log_mel.transpose(0, 1)  # (time, mel)
    return log_mel