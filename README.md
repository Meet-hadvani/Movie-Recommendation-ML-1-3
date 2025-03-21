# 🎬 Movie-Recommendation-ML-1-3

✅ **1/3 AI Projects**  
This is a collaborative filtering-based movie recommendation system using **K-Nearest Neighbors (KNN)** and the **MovieLens 100K dataset**. It is a user-based recommendation engine that identifies similar users or movies based on rating behavior.

---

## 📌 Project Highlights

- **Model:** K-Nearest Neighbors (Collaborative Filtering)
- **Data:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- **Goal:** Recommend similar movies to a given movie based on rating patterns.
- **Tech Stack:** Python, Pandas, Scikit-learn, NumPy

---

## 🔍 How It Works

- Movie ratings by users are used to create a **user-item matrix**.
- The matrix is transposed to a **movie-feature matrix**.
- **Cosine similarity** is used to measure distance between movies.
- KNN is used to find the top-N similar movies.

---

## ⚙️ Algorithm Details

### Model: `KNeighborsClassifier`
- **Metric:** Cosine Similarity
- **Algorithm:** Brute Force Search (not efficient for large-scale data)
- **K:** Number of nearest neighbors (default 5)

### Complexity:
- **Time Complexity (Training):** O(1) — Lazy learner (no real training phase)
- **Time Complexity (Query):** O(n × m) where `n` is the number of movies and `m` is features (users)
- **Space Complexity:** O(n × m) for storing the user-item matrix

---

## 📊 Dataset Summary

- **Users:** 943  
- **Movies:** 1,682  
- **Ratings:** 100,000  
- **Rating Scale:** 1 to 5

---

## 🧠 Why KNN?
- Simple to implement
- Easy to explain and visualize
- Works well for small datasets

---

## ⚠️ Limitations

- Doesn't scale to large datasets (inefficient with millions of users/movies)
- No deep understanding of content (only based on ratings)
- Cold-start problem (new users or new movies without data)
- No personalization (unless user-based KNN is used)

---

## 🚀 Ideas for Improvement

| Model Type | Description | Benefit |
|------------|-------------|---------|
| **Matrix Factorization (SVD, NMF)** | Learns latent factors between users and items | Better generalization |
| **Autoencoders** | Deep learning for dense representations | Handles sparsity |
| **CNN / RCNN** | If movie content (images, trailers) is available | Context-aware recommendations |
| **Hybrid Models** | Combine content + collaborative filtering | Handles cold-start and sparsity |

---

## 📂 Folder Structure

