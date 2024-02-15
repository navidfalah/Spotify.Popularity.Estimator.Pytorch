import pandas as pd
import Levenshtein as lev
import os

# Function to list music file names from a directory
def list_music_files(music_files_directory):
    return [f for f in os.listdir(music_files_directory) if os.path.isfile(os.path.join(music_files_directory, f))]

# Function to find the nearest music file name based on string similarity
def find_nearest_music_name(name, music_names):
    closest_name = None
    highest_similarity = 0
    for music_name in music_names:
        music_name_without_extension = os.path.splitext(music_name)[0]
        similarity = lev.ratio(name.lower(), music_name_without_extension.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_name = music_name
    return closest_name

# Main process
def process_music_data(csv_file_path, music_files_directory, target_column):
    # Load the CSV file
    try:
        music_df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return

    # Verify the target column exists
    if target_column not in music_df.columns:
        print(f"Column '{target_column}' not found in CSV. Available columns: {music_df.columns}")
        return
    
    # List music file names
    music_file_names = list_music_files(music_files_directory)
    
    # Find the nearest music file name and add it to a new column
    music_df['nearest_music_file'] = music_df.apply(lambda row: find_nearest_music_name(row[target_column], music_file_names), axis=1)
    
    # Save the updated DataFrame to a new CSV file
    updated_csv_path = csv_file_path.replace('.csv', '_updated.csv')
    music_df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved to {updated_csv_path}")

# Update these paths and column name as necessary
csv_file_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/all_spotify_data_refined.csv'
music_files_directory = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav_new'
target_column = 'name'  # Adjust based on the actual column name

process_music_data(csv_file_path, music_files_directory, target_column)
