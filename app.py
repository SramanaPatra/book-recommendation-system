import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic

st.title("📚 Personalized Book Recommendation System")

st.write("This system recommends books based on user preferences using Collaborative Filtering.")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("data/books copy.csv")
    ratings = pd.read_csv("data/ratings copy.csv", nrows=200000)

    ratings = ratings[['user_id', 'book_id', 'rating']]

    # Filter active users
    user_counts = ratings['user_id'].value_counts()
    active_users = user_counts[user_counts >= 20].index
    ratings = ratings[ratings['user_id'].isin(active_users)]

    # Filter popular books
    book_counts = ratings['book_id'].value_counts()
    popular_books = book_counts[book_counts >= 10].index
    ratings = ratings[ratings['book_id'].isin(popular_books)]

    return books, ratings

books, ratings = load_data()

valid_users = ratings['user_id'].unique()

# -------------------------------
# Train Model (Cached)
# -------------------------------
@st.cache_resource
def train_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    model = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    model.fit(trainset)
    return model

model = train_model(ratings)

st.success(f"Model loaded successfully. Active users: {len(valid_users)}")

# -------------------------------
# User Input
# -------------------------------
user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Get Recommendations"):

    if user_id not in valid_users:
        st.error("User not found in active dataset.")
        st.write("Try one of these IDs:", valid_users[:10])
    else:
        st.write("Generating recommendations...")

        rated_books = ratings[ratings['user_id'] == user_id]['book_id'].tolist()
        all_books = books['book_id'].unique()

        predictions = []

        for book_id in all_books:
            if book_id not in rated_books:
                pred = model.predict(user_id, book_id)
                predictions.append((book_id, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_books = predictions[:5]

        st.subheader("Top Recommended Books")

        for book_id, rating in top_books:
            title = books[books['book_id'] == book_id]['title'].values[0]
            st.write(f"**{title}**  (Predicted Rating: {round(rating,2)})")