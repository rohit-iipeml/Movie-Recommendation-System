📌 Movie Recommendation System
This is a simple content-based movie recommendation system that suggests movies based on user input. It uses TF-IDF vectorization and cosine similarity to find the most relevant matches.

📂 Dataset

Source: Wikipedia Movie Plots Dataset
Columns Used: Title, Plot
Processing: The dataset is cleaned, preprocessed (stopword removal, lemmatization), and transformed into TF-IDF vectors.
🛠 Setup

1. Install Python (If Not Installed)
Recommended Version: Python 3.8+
Install from python.org
2. Install Dependencies
Run the following command to install the required Python libraries:
**pip install -r requirements.txt**

🚀 Running the Recommendation System

Run the script from the command line:

python lumaa.py "I love thrilling action movies set in space, with a comedic twist."

📊 Example Output

Top Recommended Movies:

Title: Ilamai Oonjal
Plot: Some college students from Chennai go on vacation...
Similarity Score: 0.18

Title: Crayon Shin-chan: The Storm Called: Operation Rain
Plot: A girl named Lemon suddenly appears...
Similarity Score: 0.14
