import pandas as pd
import os

# Path to your CSV file
csv_file_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/spotify_data_updated.csv'
# Path to the directory where you want to check the existence of files
directory_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav'
# Column name which contains the filenames
filename_column = 'music_file'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Filter rows where the file does not exist in the specified directory
df = df[df[filename_column].apply(lambda x: os.path.exists(os.path.join(directory_path, x)))]

# Save the updated DataFrame back to CSV
df.to_csv(csv_file_path, index=False)
