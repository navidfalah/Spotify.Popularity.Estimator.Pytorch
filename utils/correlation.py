# Calculate the correlation of all features with the popularity
popularity_correlation = numeric_columns.corr()['popularity'].sort_values()
# Convert the correlation series to a Markdown table format
popularity_correlation_markdown = popularity_correlation.to_markdown()

# Return the Markdown table
popularity_correlation_markdown
