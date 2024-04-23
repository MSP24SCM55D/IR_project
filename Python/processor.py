from flask import Flask, request, jsonify
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample document collection
documents = [
    "Machine learning is the study of computer algorithms that improve automatically through experience.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
    "Python is an interpreted, high-level, general-purpose programming language.",
    "Flask is a micro web framework written in Python.",
    "Scrapy is a fast, open-source web crawling framework written in Python."
]

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

@app.route('/')
def index():
    return 'Welcome to the Flask app!'

@app.route('/query', methods=['POST'])
def process_query():
    # Parse JSON request
    request_data = request.get_json()
    query = request_data['query']
    top_k = request_data.get('top_k', 5)

    # Validate query
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Query expansion using WordNet
    expanded_query = expand_query(query)

    # Vectorize query
    query_vector = tfidf_vectorizer.transform([expanded_query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get top-K ranked results
    top_results_indices = similarity_scores.argsort()[0][-top_k:][::-1]
    top_results = [(documents[i], similarity_scores[0][i]) for i in top_results_indices]

    return jsonify({"results": top_results})

def expand_query(query):
    # WordNet query expansion
    expanded_query = []
    for word in query.split():
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        if synonyms:
            expanded_query.extend(list(synonyms))
        else:
            expanded_query.append(word)
    return ' '.join(expanded_query)

if __name__ == '__main__':
    app.config['DEBUG'] = False  # Disable default Flask route for root URL
    app.run(debug=True)
