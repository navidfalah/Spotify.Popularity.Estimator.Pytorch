import csv

# Path to the input CSV file
input_csv_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/all_spotify_data_refined_updated.csv'
# Path to the output CSV file (sorted)
output_csv_path = '/home/navid/Desktop/Spotify-Popularity-Estimator-Pytorch/data_spotify/all_spotify_data_output.csv'
# Index of the column to sort by (e.g., 0 for the first column)

COLUMN_NAME = 'music_file'

# Read the CSV file, determine the index of the specified column, and sort it
with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    # Convert reader to a list of dicts to sort it
    rows = list(reader)
    # Sort the rows based on the specified column name
    # Note: Adjust the key function for different sorting criteria or column types
    sorted_rows = sorted(rows, key=lambda row: row[COLUMN_NAME].lower())

# Write the sorted data to a new CSV file, including the header
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    # Write the header first
    writer.writeheader()
    # Write the sorted rows
    writer.writerows(sorted_rows)

print(f'CSV file has been sorted by column "{COLUMN_NAME}" and saved to {output_csv_path}.')
