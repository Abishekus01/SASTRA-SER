import os
from pydub import AudioSegment

DATASET_PATH = "datasets/TESS/english"

for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)

    for emotion in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path, emotion)

        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                wav_path = os.path.join(emotion_path, file)

                audio = AudioSegment.from_wav(wav_path)

                mp3_path = wav_path.replace(".wav", ".mp3")
                mpeg_path = wav_path.replace(".wav", ".mpeg")

                audio.export(mp3_path, format="mp3")
                audio.export(mpeg_path, format="mp3")  # MPEG = MP3 codec

print("All WAV → MP3 & MPEG converted")
