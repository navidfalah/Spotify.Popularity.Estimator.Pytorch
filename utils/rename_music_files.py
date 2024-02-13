import os

def remove_string_from_filename(directory, string):
    for filename in os.listdir(directory):
        if string in filename:
            new_filename = filename.replace(string, '')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Removed "{string}" from {filename}')

directory = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav'
string_to_remove = '(128)'

remove_string_from_filename(directory, string_to_remove)
