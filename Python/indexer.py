# indexer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class Indexer:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.index = self.vectorizer.fit_transform(documents)

    def save_index(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.index, file)

    def load_index(self, filename):
        with open(filename, 'rb') as file:
            self.index = pickle.load(file)

    def cosine_sim(self, query_vector):
        return cosine_similarity(query_vector, self.index)

# Example usage:
if __name__ == "__main__":
    # Example documents
    documents = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    # Initialize and save the index
    indexer = Indexer(documents)
    indexer.save_index("index.pkl")

    # Load the index
    loaded_indexer = Indexer(documents)
    loaded_indexer.load_index("index.pkl")

    # Example query
    query = "This is the second document."

    # Transform the query to vector
    query_vector = loaded_indexer.vectorizer.transform([query])

    # Calculate cosine similarity
    similarity_scores = loaded_indexer.cosine_sim(query_vector)
    print(similarity_scores)
