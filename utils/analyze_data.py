# Prepare a Markdown table with descriptive statistics for numerical features
stats = numeric_columns.describe().transpose()
# Select relevant statistics
relevant_stats = stats[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'})

# Convert the DataFrame to a Markdown table format
stats_markdown = relevant_stats.to_markdown()

# Return the Markdown table
stats_markdown
