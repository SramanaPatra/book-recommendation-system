# ----------------------------------------------------------
# Personalized Book Recommendation System
# Book-Based Collaborative Filtering (Cosine Similarity)
# ----------------------------------------------------------

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("Current path:", os.getcwd())
print("\nStarting Book Recommendation System (Book-Based)...\n")

# ----------------------------------------------------------
# Step 1: Load Datasets
# ----------------------------------------------------------

print("Loading datasets...")

books = pd.read_csv("data/books copy.csv")
ratings = pd.read_csv("data/ratings copy.csv", nrows=200000)

print("Books loaded:", books.shape)
print("Ratings loaded:", ratings.shape)

ratings = ratings[['user_id', 'book_id', 'rating']]

# ----------------------------------------------------------
# Step 2: Data Filtering (same as before)
# ----------------------------------------------------------

print("\nFiltering active users and popular books...")

user_counts = ratings['user_id'].value_counts()
active_users = user_counts[user_counts >= 20].index
ratings = ratings[ratings['user_id'].isin(active_users)]

book_counts = ratings['book_id'].value_counts()
popular_books = book_counts[book_counts >= 10].index
ratings = ratings[ratings['book_id'].isin(popular_books)]

print("Filtered ratings:", ratings.shape)

# ----------------------------------------------------------
# Step 3: Merge Data
# ----------------------------------------------------------

print("\nMerging datasets...")

data = pd.merge(ratings, books, left_on="book_id", right_on="book_id")

# ----------------------------------------------------------
# Step 4: Create Pivot Table (Book vs User)
# ----------------------------------------------------------

print("Creating pivot table...")

pivot = data.pivot_table(index="title", columns="user_id", values="rating").fillna(0)

print("Pivot table shape:", pivot.shape)

# ----------------------------------------------------------
# Step 5: Compute Similarity Matrix
# ----------------------------------------------------------

print("Computing similarity matrix...")

similarity = cosine_similarity(pivot)
similarity_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

print("Model ready!")

# ----------------------------------------------------------
# Step 6: Recommendation Function (Book-Based)
# ----------------------------------------------------------

def recommend_books(book_name, n=5):

    if book_name not in similarity_df.index:
        print("\nBook not found!")
        print("Try one of these:\n")
        print(list(similarity_df.index[:10]))
        return

    print(f"\nRecommendations for: {book_name}\n")

    similar_scores = similarity_df[book_name].sort_values(ascending=False)[1:n+1]

    for i, (book, score) in enumerate(similar_scores.items(), start=1):
        print(f"{i}. {book} (Similarity: {round(score, 2)})")

# ----------------------------------------------------------
# Step 7: Input Loop (Book Name instead of User ID)
# ----------------------------------------------------------

while True:
    book_input = input("\nEnter Book Name (or type 'exit'): ")

    if book_input.lower() == 'exit':
        print("Exiting program...")
        break

    recommend_books(book_input, 5)