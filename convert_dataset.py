import os
from pydub import AudioSegment
from pydub.utils import which

# 🔧 Fix: point pydub to ffmpeg
AudioSegment.converter = which("ffmpeg")

DATASET_PATH = "datasets/TESS/english"

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            wav_path = os.path.join(root, file)

            audio = AudioSegment.from_wav(wav_path)

            mp3_path = wav_path.replace(".wav", ".mp3")
            ogg_path = wav_path.replace(".wav", ".ogg")

            audio.export(mp3_path, format="mp3")
            audio.export(ogg_path, format="ogg")

print("Conversion completed: WAV → MP3 & OGG")