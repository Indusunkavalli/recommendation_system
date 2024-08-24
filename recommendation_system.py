import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample movie ratings data
ratings_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5],
    'movie_id': [101, 102, 101, 103, 102, 104, 103, 105, 104],
    'rating': [5, 3, 4, 2, 4, 5, 2, 4, 3]
}

# Sample movie data
movies_data = {
    'movie_id': [101, 102, 103, 104, 105],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genre': ['Action', 'Action', 'Comedy', 'Comedy', 'Drama']
}

ratings_df = pd.DataFrame(ratings_data)
movies_df = pd.DataFrame(movies_data)

# Collaborative Filtering

# Create a user-item matrix
user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_movies_collaborative(user_id, user_item_matrix, user_similarity_df, top_n=3):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users.index != user_id]
    similar_users = similar_users.head(top_n).index
    recommendations = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
    recommendations = recommendations[user_item_matrix.loc[user_id] == 0]  # Exclude already rated movies
    return recommendations.head(top_n)

# Content-Based Filtering

# TF-IDF Vectorizer for genres
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genre'])

def recommend_movies_content_based(movie_title, movies_df, tfidf_matrix, top_n=3):
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1]
    return movies_df.iloc[similar_indices][['title', 'genre']]

# Example Usage
if __name__ == "__main__":
    # Collaborative Filtering Recommendations
    user_id = 1
    print("Collaborative Recommendations for User 1:")
    collaborative_recommendations = recommend_movies_collaborative(user_id, user_item_matrix, user_similarity_df)
    print(collaborative_recommendations)

    # Content-Based Filtering Recommendations
    movie_title = 'Movie A'
    print("\nContent-Based Recommendations for 'Movie A':")
    content_based_recommendations = recommend_movies_content_based(movie_title, movies_df, tfidf_matrix)
    print(content_based_recommendations)
