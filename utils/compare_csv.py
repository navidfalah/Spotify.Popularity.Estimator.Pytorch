import pandas as pd

# Assuming paths to your CSV files
popularity_csv_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/songs_popularity.csv'  # The path to your existing songs popularity CSV
additional_data_csv_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/spotify_data_updated.csv'  # The path to your new CSV with additional data

# Load the CSV files into pandas DataFrames
popularity_df = pd.read_csv(popularity_csv_path)
additional_data_df = pd.read_csv(additional_data_csv_path)

# You might need to preprocess the columns used for merging (e.g., lowercasing, removing file extensions)
# For simplicity, let's assume we are merging on the "Song Title" from the first CSV
# and a transformed version of "music_file" from the second CSV

# Transforming the music_file column to match the Song Title format for merging might look like this:
additional_data_df['merge_key'] = additional_data_df['music_file'].apply(lambda x: x.replace(' (128).wav', ''))

# Merging the DataFrames on the song titles
merged_df = pd.merge(popularity_df, additional_data_df, left_on='Song Title', right_on='merge_key', how='left')

# Saving the merged DataFrame to a new CSV file
merged_csv_path = 'merged_songs_data.csv'
merged_df.to_csv(merged_csv_path, index=False)

print("Merging complete. The merged data is saved to:", merged_csv_path)
