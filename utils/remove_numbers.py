import os

# Replace 'your/directory/path' with the actual path to your files
directory_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav'

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.wav') and filename[0].isdigit():
        # Split the filename on the first occurrence of ". "
        split_name = filename.split('. ', 1)
        if len(split_name) > 1:
            # Construct the new filename without the leading number and period
            new_filename = split_name[1]
            # Construct the full old and new file paths
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_filename}"')
