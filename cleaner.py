import pandas as pd
import os
import re

# Define your CSV file and audio directory paths
ANNOTATIONS_FILE = "/home/navid/Desktop/data_spotify/songs.csv"
AUDIO_DIR = "/home/navid/Desktop/data_spotify/wav"

# Read the CSV file
df = pd.read_csv(ANNOTATIONS_FILE)

# Iterate through the DataFrame and check if corresponding audio files exist
records_to_delete = []
for index, row in df.iterrows():
    # Sanitize the name by removing text within parentheses
    sanitized_name = re.sub(r'\s*\(.*?\)\s*', ' ', row['name']).strip()

    # Check if any file in the directory contains the sanitized name
    file_exists = any(sanitized_name in file for file in os.listdir(AUDIO_DIR) if file.endswith('.wav'))

    if not file_exists:
        records_to_delete.append(index)

# Drop the records where audio files do not exist
df = df.drop(records_to_delete)

# Save the updated DataFrame back to CSV
df.to_csv(ANNOTATIONS_FILE, index=False)
