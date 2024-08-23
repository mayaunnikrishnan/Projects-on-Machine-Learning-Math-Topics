import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
anime_df = pd.read_csv("anime.csv")

# Fill missing values
anime_df['genre'] = anime_df['genre'].fillna('')
anime_df['type'] = anime_df['type'].fillna('')
anime_df['rating']=anime_df['rating'].fillna(anime_df['rating'].mean())

# Combine text features
anime_df['combined_features'] = anime_df['genre'] + ' ' + anime_df['type'] + ' ' + anime_df['rating'].astype(str)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['combined_features'])

def recommend_anime(anime_title, tfidf_matrix=tfidf_matrix):
    # Find the index of the given anime title
    anime_index = anime_df[anime_df['name'] == anime_title].index[0]

    # Calculate cosine similarity between the given anime and all other anime
    cosine_similarities = linear_kernel(tfidf_matrix[anime_index], tfidf_matrix).flatten()

    # Get top 10 similar anime indices
    similar_anime_indices = cosine_similarities.argsort()[-11:-1][::-1]

    # Get top 10 similar anime titles
    similar_anime_titles = anime_df.iloc[similar_anime_indices]['name'].values

    return similar_anime_titles

# Streamlit UI
st.title("Anime Recommendation System")

# Apply custom CSS to change the color of the text input
st.markdown(
    """
    <style>
    .st-ef {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input field for anime title
anime_title = st.text_input("Enter an anime title:", "")

# Recommend button
if st.button("Recommend"):
    if anime_title:
        recommended_anime = recommend_anime(anime_title)
        st.subheader(f"Anime similar to {anime_title}:")
        for anime in recommended_anime:
            st.write(anime)
    else:
        st.write("Please enter an anime title.")
