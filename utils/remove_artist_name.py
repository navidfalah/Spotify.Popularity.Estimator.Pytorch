import os
import re

# List of artist names to remove from file names
artists_to_remove = [
    "J. Cole", "Cardi B", "Juice WRLD", "d4vd", "Lewis Capaldi", "Myke Towers",
    "070 Shake", "Ava Max", "Britney Spears", "Pitbull", "Jason Derulo", "Tate McRae",
    "The Neighbourhood", "Ellie Goulding", "Miguel", "James Arthur", "Stephen Sanchez",
    "Avicii", "The Chainsmokers", "Maluma", "Lil Uzi Vert", "Quevedo", "Linkin Park",
    "Michael Jackson", "JAY-Z", "Manuel Turizo", "Oliver Tree", "Olivia Rodrigo", "P!nk",
    "Feid", "Robin Schulz", "XXXTENTACION", "One Direction", "Charlie Puth", "RAYE",
    "BTS", "Lil Nas X", "Meghan Trainor", "Rema", "Lana Del Rey", "KAROL G", "Camila Cabello",
    "Black Eyed Peas", "ROSALÍA", "Arctic Monkeys", "Halsey", "Marshmello", "Rauw Alejandro",
    "Kendrick Lamar", "Nicki Minaj", "Queen", "Travis Scott", "The Kid LAROI", "Shawn Mendes",
    "Sia", "Tiësto", "Future", "Kim Petras", "Ozuna", "Daddy Yankee", "Elton John",
    "OneRepublic", "J Balvin", "Katy Perry", "Khalid", "Bebe Rexha", "Adele", "Billie Eilish",
    "Post Malone", "Kanye West", "Metro Boomin", "Doja Cat", "Beyoncé", "Maroon 5",
    "Chris Brown", "Selena Gomez", "Ariana Grande", "Bizarrap", "Lady Gaga", "Imagine Dragons",
    "Bruno Mars", "Calvin Harris", "Coldplay", "21 Savage", "Bad Bunny", "Harry Styles",
    "Eminem", "Justin Bieber", "Drake", "SZA", "David Guetta", "Sam Smith", "Rihanna",
    "Shakira", "Taylor Swift", "Miley Cyrus", "Ed Sheeran", "The Weeknd"
]

# Replace 'your/directory/path' with the actual path to your files
directory_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/wav'


def clean_filename(filename, artists):
    # Extract the base name and extension
    name, ext = os.path.splitext(filename)
    
    # Replace underscores with spaces in the base name
    name = name.replace('_', ' ')
    
    # Initial cleaning to remove leading spaces from the base name
    cleaned_name = name.lstrip()
    
    # Remove artist names
    for artist in artists:
        pattern = re.compile(re.escape(artist), re.IGNORECASE)
        cleaned_name = pattern.sub('', cleaned_name)
    
    # Remove parentheses and dashes from the base name
    cleaned_name = re.sub(r'[\(\)-]', '', cleaned_name)
    
    # Remove instances of "feat" and its variations from the base name
    cleaned_name = re.sub(r'\s*\(?\bfeat\.?\b.*\)?', '', cleaned_name, flags=re.IGNORECASE)
    
    # Remove extra spaces from the base name
    cleaned_name = " ".join(cleaned_name.split())
    
    # Construct the new filename with the original extension
    new_filename = cleaned_name + ext
    
    return new_filename

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.wav'):  # Only process .wav files
        new_filename = clean_filename(filename, artists_to_remove)
        # Check if changes were made to the filename
        if new_filename.lower() != filename.lower():
            # Construct the full old and new file paths
            old_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_filename}"')
            