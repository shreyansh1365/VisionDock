import wave
import math
import struct
import os

# Ensure the assets folder exists
os.makedirs("assets", exist_ok=True)

def create_sound(filename, frequency, duration, volume=0.5):
    sample_rate = 44100
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(2) # Stereo
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            # Generate sine wave math
            value = int(volume * 32767.0 * math.sin(frequency * math.pi * 2 * i / sample_rate))
            # Pack as binary data for Left and Right channels
            data = struct.pack('<hh', value, value) 
            wav_file.writeframesraw(data)

print("Generating sound files...")

# 1. Create a sharp, high-pitched beep (1000 Hz, 0.1 seconds)
create_sound("assets/beep.wav", 1000, 0.1, 0.8)

# 2. Create a low, gentle heartbeat thump (200 Hz, 0.3 seconds)
create_sound("assets/heartbeat.wav", 200, 0.3, 0.4)

print("✅ Success! Valid audio files have been saved to the 'assets' folder.")