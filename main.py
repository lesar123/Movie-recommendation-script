import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define file paths for movies and ratings
movies_path = '/home/doc123/Downloads/ml-25m/movies.csv'
ratings_path = '/home/doc123/Downloads/ml-25m/ratings.csv'

# Load movies data
movies = pd.read_csv(movies_path)

# Display the first few rows to verify the structure
print(movies.head())

# Select only necessary columns for recommendation
# Split genres into a single 'genres' column and clean the title column
movies['genres'] = movies['genres'].fillna('')
movies['title'] = movies['title'].fillna('').apply(lambda x: re.sub(r'\(\d{4}\)', '', x).strip().lower())

# Add example descriptions for testing purposes (since the dataset does not include descriptions)
descriptions = [
    "A man finds redemption through a lifelong friendship.",
    "A young boy navigates life under a ruthless crime family.",
    "A heroic journey through mystical lands to stop a dark wizard.",
    "Two detectives hunt down a serial killer in a disturbing tale of justice.",
    # Add more sample descriptions to match the number of movies
]
# Limit the movies dataset to match the number of descriptions
movies = movies.head(50)

# Extend descriptions to match the movies count exactly
extended_descriptions = (descriptions * (len(movies) // len(descriptions) + 1))[:len(movies)]
movies['description'] = extended_descriptions

# Use TF-IDF to vectorize descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on cosine similarity
def recommend_movies(title, cosine_sim=cosine_sim):
    title = re.sub(r'\(\d{4}\)', '', title).strip().lower()  # Clean the title input

    # Check if title exists in the database
    if title not in movies['title'].values:
        return f"Sorry, the movie '{title}' is not in our database."

    # Find the index of the movie title
    idx = movies.index[movies['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Main program for user input
if __name__ == "__main__":
    while True:
        user_input = input("Enter 'available' if you want to see the movies we have listed, or type a movie title for recommendations: ")

        if user_input.lower() == 'available':
            print("Available movies in our database:")
            print("\n".join(movies['title'].tolist()))

        elif user_input.lower() == 'exit':
            print("Goodbye!")
            break

        else:
            recommendations = recommend_movies(user_input)
            print("Recommended Movies:")
            if isinstance(recommendations, list):
                for movie in recommendations:
                    print(f"- {movie}")
            else:
                print(recommendations)
