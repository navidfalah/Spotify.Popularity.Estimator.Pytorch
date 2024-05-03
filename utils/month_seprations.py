# Extract the release month and year from the release_date column
spotify_data_predicted_csv_df['release_month'] = pd.to_datetime(spotify_data_predicted_csv_df['release_date'], errors='coerce').dt.month
spotify_data_predicted_csv_df['release_year'] = pd.to_datetime(spotify_data_predicted_csv_df['release_date'], errors='coerce').dt.year

# Drop rows with NaN values in release_month and release_year after coercion
spotify_data_predicted_csv_df = spotify_data_predicted_csv_df.dropna(subset=['release_month', 'release_year'])

# Group the data by release month and year, then calculate the average popularity for each group
monthly_popularity = spotify_data_predicted_csv_df.groupby(['release_year', 'release_month'])['popularity'].mean().reset_index()
# Pivot the data to compare the same month across different years
monthly_popularity_pivot = monthly_popularity.pivot(index='release_month', columns='release_year', values='popularity').reset_index()

# Return the pivoted data for comparison
monthly_popularity_pivot