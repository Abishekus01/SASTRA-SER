import torch
import torchaudio

# Newer torchaudio versions don't need set_audio_backend. 
# It uses the best available (soundfile or sox) automatically.

SAMPLE_RATE = 16000
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def load_wav(path):
    try:
        # Load the file
        waveform, sr = torchaudio.load(path)

        # Resample if the source sample rate doesn't match our target
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to Mono by averaging channels if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Efficient Pad/Trim to exactly 3 seconds (48,000 samples)
        length = waveform.shape[0]
        if length < SAMPLES:
            pad_size = SAMPLES - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:SAMPLES]

        return waveform
    except Exception as e:
        # If a file is corrupted, return a silent tensor so training doesn't stop
        return torch.zeros(SAMPLES)