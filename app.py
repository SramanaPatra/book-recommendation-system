import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("📚 Book Recommendation System")
st.write("Get recommendations based on your favorite book!")

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

# -------------------------------
# Prepare Data (Pivot Table)
# -------------------------------
@st.cache_data
def create_pivot(books, ratings):
    data = pd.merge(ratings, books, left_on="book_id", right_on="book_id")
    pivot = data.pivot_table(index="title", columns="user_id", values="rating").fillna(0)
    return pivot

pivot = create_pivot(books, ratings)

# -------------------------------
# Compute Similarity
# -------------------------------
@st.cache_data
def compute_similarity(pivot):
    similarity = cosine_similarity(pivot)
    return pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

similarity_df = compute_similarity(pivot)

st.success("Model loaded successfully!")

# -------------------------------
# Book Selection (Dropdown)
# -------------------------------
book_list = sorted(pivot.index.tolist())
selected_book = st.selectbox("Select your favorite book:", book_list)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_books(book_name, n=5):
    similar_scores = similarity_df[book_name].sort_values(ascending=False)[1:n+1]
    return similar_scores

# -------------------------------
# Button
# -------------------------------
if st.button("Get Recommendations"):

    st.write(f"### Recommendations for: {selected_book}")

    recommendations = recommend_books(selected_book)

    for i, (book, score) in enumerate(recommendations.items(), start=1):
        st.write(f"**{i}. {book}**  (Similarity: {round(score,2)})")