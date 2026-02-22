import os
from pydub import AudioSegment

# ⚠️ DO NOT set converter path in Linux/Codespaces (ffmpeg already installed)

INPUT_DATASET = "datasets"
OUTPUT_DATASET = "datasets_converted"

TARGET_SAMPLE_RATE = 22050
SUPPORTED_FORMATS = (".wav", ".mp3", ".ogg", ".mpeg")

os.makedirs(OUTPUT_DATASET, exist_ok=True)

audio_files = []

# 🔍 Collect files
for root, dirs, files in os.walk(INPUT_DATASET):
    for file in files:
        if file.lower().endswith(SUPPORTED_FORMATS):
            audio_files.append(os.path.join(root, file))

print(f"\n🎧 Total files found: {len(audio_files)}\n")

# 🔁 Convert
for i, file_path in enumerate(audio_files):
    try:
        print(f"Processing {i+1}/{len(audio_files)}: {file_path}")

        # ✅ Auto detect format (FIX)
        audio = AudioSegment.from_file(file_path)

        # Convert to mono + target SR
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

        # Maintain folder structure
        relative_path = os.path.relpath(file_path, INPUT_DATASET)
        base_name = os.path.splitext(relative_path)[0]

        wav_output_path = os.path.join(OUTPUT_DATASET, base_name + ".wav")
        os.makedirs(os.path.dirname(wav_output_path), exist_ok=True)

        # Export only WAV for training
        audio.export(wav_output_path, format="wav")

    except Exception as e:
        print(f"❌ Skipped (corrupted or invalid): {file_path}")
        print(f"   Reason: {e}")

print("\n✅ Conversion completed!")