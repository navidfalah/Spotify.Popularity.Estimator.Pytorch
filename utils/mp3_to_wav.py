from pydub import AudioSegment
import os

# Directory containing the MP3 files
input_directory = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/mp3'

# Directory where the WAV files will be saved
output_directory = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav_new'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.mp3'):
        try:
            # Full path to the current MP3 file
            mp3_path = os.path.join(input_directory, file_name)
            
            # Name of the WAV file to be saved
            wav_name = os.path.splitext(file_name)[0] + '.wav'
            # Full path where the WAV file will be saved
            wav_path = os.path.join(output_directory, wav_name)
            
            # Load the MP3 file
            audio = AudioSegment.from_mp3(mp3_path)
            
            # Export the audio to WAV format
            audio.export(wav_path, format="wav")
            print(f'Converted {file_name} to WAV format.')
        except Exception as e:
            print(f'Failed to convert {file_name}: {e}')
