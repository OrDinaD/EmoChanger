import os
import numpy as np
import soundfile as sf

def create_test_audio(filename, duration=1, sr=16000):
    """Creates a test audio file with given duration and sample rate"""
    # Generate random noise
    audio = np.random.randn(int(duration * sr))
    # Normalize
    audio = audio / np.max(np.abs(audio))
    # Save to file
    sf.write(filename, audio, sr)
    print(f"Created {filename}")

os.makedirs('data/raw', exist_ok=True)

# Create 5 test audio files for each emotion
for i in range(5):
    create_test_audio(f'data/raw/sample{i+1}.wav', duration=1) 