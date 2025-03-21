import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=ratings_cols, encoding='latin-1')
movies = pd.read_csv("ml-100k/u.item", sep="|", names=movies_cols, usecols=[0, 1], encoding='latin-1')

data = pd.merge(ratings, movies, on="movie_id")
movie_matrix = data.pivot_table(index = 'user_id', columns='title', values='rating')
movie_matrix_filled = movie_matrix.fillna(0)

movie_features = movie_matrix_filled.T

#KNN model
model = NearestNeighbors(metric = 'cosine', algorithm='brute')
model.fit(movie_features)

def recommend_movies(movie_title, n_recommendations=5):
    if movie_title not in movie_features.index:
        return f"Movie '{movie_title}' not found in dataset."
    
    movie_index = movie_features.index.get_loc(movie_title)
    distances, indices = model.kneighbors([movie_features.iloc[movie_index]], 
                                              n_neighbors=n_recommendations+1)
    recommended_titles = [movie_features.index[i] for i in indices.flatten() 
                          if movie_features.index[i] != movie_title]
    return recommended_titles[:n_recommendations]


if __name__ == "__main__":
    movie_name = "Star Wars (1977)"
    recommendations = recommend_movies(movie_name, 5)
    print(f"Because you liked '{movie_name}', you may also like:")
    for rec in recommendations:
        print(f"- {rec}")