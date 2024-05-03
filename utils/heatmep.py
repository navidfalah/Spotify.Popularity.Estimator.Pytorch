# Regenerate the heatmap to visualize the popularity trends
plt.figure(figsize=(20, 10))
sns.heatmap(monthly_popularity_pivot.set_index('release_month'), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Heatmap of Average Popularity by Release Month and Year')
plt.xlabel('Release Year')
plt.ylabel('Release Month')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
# Save the heatmap again
plt.savefig('/app/static/uploads/popularity_heatmap.png')
plt.close()

# Provide the path to the saved heatmap
'/app/static/uploads/popularity_heatmap.png'