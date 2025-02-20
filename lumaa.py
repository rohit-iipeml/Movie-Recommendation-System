# -*- coding: utf-8 -*-
"""Lumaa.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cVhHXlQwvR_xd7FAdgV4hHoA_gH5MamQ
"""

import pandas as pd

df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df[['Title', 'Plot']]
df.dropna(inplace=True)
df.head()

df.shape

df_sample = df.sample(n=500, random_state=42)  # Select 500 random movies
df_sample.reset_index(drop=True, inplace=True)
df_sample.shape

df_sample

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

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

df_sample['Cleaned_Plot'] = df_sample['Plot'].apply(preprocess_text)
df_sample

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_sample['Cleaned_Plot'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

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

# Sample user input
user_input = "I love thrilling action movies set in space, with a comedic twist."
recommended_movies = recommend_movies(user_input, df_sample, vectorizer, tfidf_matrix, top_n=5)
recommended_movies

