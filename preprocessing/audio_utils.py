import torch
import torchaudio

# Set backend to soundfile for max efficiency with WAV files
torchaudio.set_audio_backend("soundfile")

SAMPLE_RATE = 16000
SAMPLES = 16000 * 3  # 3 seconds

def load_wav(path):
    try:
        # Load and immediately move to CPU/GPU memory
        waveform, sr = torchaudio.load(path)
        
        # Resample only if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        
        # Mono conversion
        waveform = waveform.mean(0)
        
        # Fast Pad/Trim
        length = waveform.shape[0]
        if length < SAMPLES:
            waveform = torch.cat([waveform, torch.zeros(SAMPLES - length)])
        else:
            waveform = waveform[:SAMPLES]
            
        return waveform
    except Exception:
        return torch.zeros(SAMPLES)