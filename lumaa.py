# -*- coding: utf-8 -*-
"""Lumaa - Movie Recommendation System"""

import pandas as pd
import argparse
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df[['Title', 'Plot']]
df.dropna(inplace=True)

# Sample 500 random movies
df_sample = df.sample(n=500, random_state=42)
df_sample.reset_index(drop=True, inplace=True)

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Lowercasing
    - Removing special characters
    - Tokenizing
    - Removing stopwords
    - Lemmatizing
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# Apply text preprocessing to movie plots
df_sample['Cleaned_Plot'] = df_sample['Plot'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_sample['Cleaned_Plot'])
print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")

# Recommendation function
def recommend_movies(user_query, df, vectorizer, tfidf_matrix, top_n=5):
    """
    Recommend movies based on cosine similarity with the user's input.

    Parameters:
        user_query (str): User's text input.
        df (DataFrame): DataFrame containing movie titles and cleaned plots.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): TF-IDF transformed movie plots.
        top_n (int): Number of recommendations to return.

    Returns:
        DataFrame: Top N recommended movies with similarity scores.
    """
    user_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    recommended_movies = df.iloc[top_indices][['Title', 'Plot']].copy()
    recommended_movies['Similarity Score'] = similarity_scores[top_indices]

    return recommended_movies

# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("query", type=str, help="User query for movie recommendations")
    args = parser.parse_args()

    # Get recommendations
    recommendations = recommend_movies(args.query, df_sample, vectorizer, tfidf_matrix, top_n=5)

    # Print output in a readable format
    print("\n🎬 **Top Recommended Movies:**\n")
    for i, row in recommendations.iterrows():
        print(f"🔹 **Title:** {row['Title']}")
        print(f"📖 **Plot:** {row['Plot'][:300]}...")  # Show first 300 characters of the plot
        print(f"⭐ **Similarity Score:** {row['Similarity Score']:.4f}\n")
