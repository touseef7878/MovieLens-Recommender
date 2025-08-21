# MovieLens 100k Recommendation System
# 📌 Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

# Step 1: Load Dataset
# Load user ratings
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# Load movie info
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None,
                     names=["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
                            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])

# Merge ratings with movies
data = pd.merge(ratings, movies, left_on="item_id", right_on="movie_id")

print("Dataset Loaded ✅")
print(data.head())

# Step 2: Create User-Movie Rating Matrix
user_movie_matrix = data.pivot_table(index="user_id", columns="title", values="rating")

# Step 3: User-Based Collaborative Filtering
# Compute similarity between users
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies_user(user_id, num_recommendations=5):
    # Get similarity scores
    sim_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Weighted average of ratings from similar users
    sim_scores = []
    for other_user in sim_users:
        common_movies = user_movie_matrix.loc[[user_id, other_user]].dropna(axis=1, how="any")
        if len(common_movies.columns) > 0:
            sim_scores.append((other_user, user_similarity_df.loc[user_id, other_user]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Take top N similar users
    top_users = [u for u, score in sim_scores[:10]]
    recommendations = user_movie_matrix.loc[top_users].mean().sort_values(ascending=False)
    
    # Remove already watched movies
    watched = user_movie_matrix.loc[user_id].dropna().index
    final_recs = recommendations.drop(watched).head(num_recommendations)
    
    return final_recs

print("\n🎬 User-Based Recommendations for User 1:")
print(recommend_movies_user(1))

# Step 4: Item-Based Collaborative Filtering
# Compute similarity between movies
item_similarity = cosine_similarity(user_movie_matrix.fillna(0).T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def recommend_movies_item(movie_title, num_recommendations=5):
    similar_items = item_similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations+1]
    return similar_items

print("\n🎬 Item-Based Recommendations for 'Toy Story (1995)':")
print(recommend_movies_item("Toy Story (1995)"))

# Step 5 (Bonus): Matrix Factorization (SVD)
# Convert to matrix
R = user_movie_matrix.fillna(0).values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Apply SVD
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# Predict ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
pred_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

def recommend_movies_svd(user_id, num_recommendations=5):
    sorted_ratings = pred_df.loc[user_id].sort_values(ascending=False)
    watched = user_movie_matrix.loc[user_id].dropna().index
    recommendations = sorted_ratings.drop(watched).head(num_recommendations)
    return recommendations

print("\n🎬 SVD Recommendations for User 1:")
print(recommend_movies_svd(1))
