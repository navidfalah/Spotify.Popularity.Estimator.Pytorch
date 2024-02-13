import os
import subprocess

path = '/home/navid/Desktop/mp3_songs'  # Replace with the path to your MP3 files
output_path = '/home/navid/Desktop/wav'  # Replace with the path where you want to save WAV files

if not os.path.exists(output_path):
    os.makedirs(output_path)

for file in os.listdir(path):
    if file.endswith('.mp3'):
        mp3_file_path = os.path.join(path, file)
        wav_file_path = os.path.join(output_path, file[:-4] + '.wav')
        subprocess.run(['ffmpeg', '-i', mp3_file_path, wav_file_path])
