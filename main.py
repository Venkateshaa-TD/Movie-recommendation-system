import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print("✅ Script is running...")

# Load the dataset
movies = pd.read_csv("movies.csv")
print("✅ Loaded movies.csv with shape:", movies.shape)

# Clean up genres column (replace | with space)
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
print("✅ Created TF-IDF matrix with shape:", tfidf_matrix.shape)

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("✅ Computed cosine similarity matrix.")

# Create reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommender function
def get_recommendations(title, num_recommendations=3):
    if title not in indices:
        return ["❌ Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# --- Ask the user for a movie ---
print("\n🔍 Enter a movie title exactly as in dataset (e.g., Toy Story (1995))")
user_input = input("🎬 Movie title: ")

# Get recommendations
recommendations = get_recommendations(user_input)

# Show results
print("\n✅ Recommended Movies:")
for rec in recommendations:
    print("👉", rec)
