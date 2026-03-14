# ---------------------------------------------
# Personalized Book Recommendation System
# User-Based Collaborative Filtering using KNN
# ---------------------------------------------

# Step 1: Import libraries
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate

print("Starting Book Recommendation System...\n")

# ---------------------------------------------
# Step 2: Load Dataset
# ---------------------------------------------

print("Loading datasets...")

books = pd.read_csv("data/books.csv")
ratings = pd.read_csv("data/ratings.csv")

# If system is slow, use this instead:
# ratings = pd.read_csv("data/ratings.csv", nrows=500000)

print("Books loaded:", books.shape)
print("Ratings loaded:", ratings.shape)

# Keep only required columns
ratings = ratings[['user_id', 'book_id', 'rating']]

# Remove missing values if any
ratings.dropna(inplace=True)

# ---------------------------------------------
# Step 3: Prepare data for Surprise library
# ---------------------------------------------

print("\nPreparing data for model...")

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    ratings[['user_id', 'book_id', 'rating']],
    reader
)

trainset = data.build_full_trainset()

# ---------------------------------------------
# Step 4: Build KNN Model
# ---------------------------------------------

print("Training KNN model...")

sim_options = {
    'name': 'cosine',      # similarity measure
    'user_based': True     # user-user similarity
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# ---------------------------------------------
# Step 5: Evaluate Model
# ---------------------------------------------

print("\nEvaluating model (RMSE)...")
cross_validate(model, data, measures=['RMSE'], cv=3, verbose=True)

# ---------------------------------------------
# Step 6: Recommendation Function
# ---------------------------------------------

def recommend_books(user_id, n=5):
    print(f"\nGenerating recommendations for User {user_id}...")

    # Books already rated by this user
    rated_books = ratings[ratings['user_id'] == user_id]['book_id'].tolist()

    # All book ids
    all_books = books['book_id'].unique()

    predictions = []

    # Predict ratings for books not rated yet
    for book_id in all_books:
        if book_id not in rated_books:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))

    # Sort by predicted rating (highest first)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Top N recommendations
    top_books = predictions[:n]

    print("\nTop Recommended Books:\n")

    for book_id, rating in top_books:
        title = books[books['book_id'] == book_id]['title'].values[0]
        print(f"{title}  | Predicted Rating: {round(rating, 2)}")

# ---------------------------------------------
# Step 7: User Input Loop
# ---------------------------------------------

while True:
    try:
        user_input = int(input("\nEnter User ID (1–50000) or 0 to exit: "))

        if user_input == 0:
            print("Exiting program...")
            break

        recommend_books(user_input, 5)

    except:
        print("Invalid input. Please enter a valid number.")
from surprise import Dataset, Reader, KNNBasic
print("Surprise installed successfully")