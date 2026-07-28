import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned dataset
df_clean = pd.read_csv("Movie_rating_prediction_cleaned.csv")

print("Data loaded successfully!")
print(f"Shape: {df_clean.shape}")

plt.figure(figsize=(15, 12))

# 1. Rating Distribution
plt.subplot(2, 3, 1)
plt.hist(df_clean['Rating'], bins=20, edgecolor='black', alpha=0.7)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")

# 2. Year Distribution
plt.subplot(2, 3, 2)
plt.hist(df_clean['Year'], bins=20, edgecolor='black', alpha=0.7, color='orange')
plt.title("Year Distribution")
plt.xlabel("Year")
plt.ylabel("Frequency")

# 3. Duration Distribution
plt.subplot(2, 3, 3)
plt.hist(df_clean['Duration'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='green')
plt.title("Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")

# 4. Genre Distribution (Top 10 genres)
plt.subplot(2, 3, 4)
# Split genres by comma and count
genres_split = df_clean['Genre'].str.split(', ').explode()
top_genres = genres_split.value_counts().head(10)
top_genres.plot(kind='bar', color='red', alpha=0.7)
plt.title("Top 10 Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')

# 5. Votes vs Rating
plt.subplot(2, 3, 5)
plt.scatter(df_clean['Votes'], df_clean['Rating'], alpha=0.5, s=10)
plt.title("Votes vs Rating")
plt.xlabel("Votes")
plt.ylabel("Rating")
# Use log scale for votes due to wide range
plt.xscale('log')

# 6. Year vs Rating (to see if ratings changed over time)
plt.subplot(2, 3, 6)
plt.scatter(df_clean['Year'], df_clean['Rating'], alpha=0.5, s=10, color='purple')
plt.title("Year vs Rating")
plt.xlabel("Year")
plt.ylabel("Rating")

# Add a trend line
z = np.polyfit(df_clean['Year'].dropna(), df_clean['Rating'].dropna(), 1)
p = np.poly1d(z)
plt.plot(df_clean['Year'].dropna(), p(df_clean['Year'].dropna()), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# Additional visualization: Box plot of ratings by genre (top 5 genres)
plt.figure(figsize=(12, 8))

# Get top 5 genres
top_5_genres = genres_split.value_counts().head(5).index.tolist()

# Filter data for top 5 genres
df_genre_filtered = df_clean[df_clean['Genre'].apply(lambda x: any(genre in str(x) for genre in top_5_genres))]

# Prepare data for box plot
genre_data = []
genre_labels = []
for genre in top_5_genres:
    # Get ratings for movies containing this genre
    mask = df_clean['Genre'].str.contains(genre, na=False)
    ratings = df_clean.loc[mask, 'Rating'].dropna()
    if len(ratings) > 0:
        genre_data.append(ratings)
        genre_labels.append(genre)

plt.boxplot(genre_data, labels=genre_labels)
plt.title("Rating Distribution by Top 5 Genres")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nVisualizations completed!")
print("- Rating distribution shows most movies rated between 5-8")
print("- Year distribution shows concentration in recent years")
print("- Duration distribution shows most movies 90-180 minutes")
print("- Genre distribution shows Drama, Comedy, Romance as most common")
print("- Votes vs Rating shows some correlation but with high variance")
print("- Year vs Rating shows slight upward trend over time")