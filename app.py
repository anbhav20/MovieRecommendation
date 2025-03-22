import os
import logging
import requests
import aiohttp
import asyncio
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from joblib import load
from aiohttp import ClientTimeout
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Setup logging with a formatted output
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load pre-trained model
MODEL_DIR = "models"
try:
    logging.info("🔄 Loading models...")
    knn_model = load(os.path.join(MODEL_DIR, "knn_model.joblib"))
    vectorizer = load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    logging.info("✅ Models loaded successfully!")
except Exception as e:
    logging.error(f"⚠️ Error loading models: {e}")
    knn_model, vectorizer = None, None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Genre mapping for movie search (TMDB genre IDs)
genre_keywords = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751,
    "fantasy": 14, "history": 36, "horror": 27, "music": 10402,
    "mystery": 9648, "romance": 10749, "science fiction": 878,
    "tv movie": 10770, "thriller": 53, "war": 10752, "western": 37
}

# Import additional functions if available
try:
    from train_model import get_movie_full_details, get_full_recommendations
except ImportError as e:
    logging.error("❌ Could not import train_model functions: " + str(e))
    get_movie_full_details = None
    get_full_recommendations = None

# Fuzzy search function for better movie name matching (for future use if dataset available)
def fuzzy_match_movie(movie_name, movie_list):
    match, score = process.extractOne(movie_name, movie_list, scorer=fuzz.partial_ratio)
    return match if score > 70 else None

def recommend(movie_name, k=9):
    """
    Recommend movies based on the movie_name.
    If a custom recommendation function is available, it is used.
    Otherwise, fallback to TMDB's recommendations endpoint.
    """
    # Use custom recommendation function if available
    if get_full_recommendations:
        try:
            rec = asyncio.run(get_full_recommendations(movie_name, k))
            return rec
        except Exception as e:
            logging.error(f"Error in get_full_recommendations: {e}")
            return {"error": "Failed to fetch recommendations."}, 500

    # Fallback: Use TMDB's recommendation API
    API_KEY = os.getenv("TMDB_API_KEY")
    if not API_KEY:
        return {"error": "TMDB API key is missing."}, 500

    BASE_URL = "https://api.themoviedb.org/3"
    search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_name}"
    search_response = requests.get(search_url)
    if search_response.status_code != 200:
        return {"error": "Failed to fetch movie info."}, 500

    search_data = search_response.json()
    if not search_data.get("results"):
        return {"error": "Movie not found."}, 404

    movie_id = search_data["results"][0]["id"]
    rec_url = f"{BASE_URL}/movie/{movie_id}/recommendations?api_key={API_KEY}"
    rec_response = requests.get(rec_url)
    if rec_response.status_code != 200:
        return {"error": "Failed to fetch recommendations."}, 500

    rec_data = rec_response.json()
    recommendations = rec_data.get("results", [])[:k]
    return recommendations

async def get_ott_links(movie_name):
    """
    Fetch OTT (streaming) provider details for a given movie from TMDB.
    """
    API_KEY = os.getenv("TMDB_API_KEY")
    BASE_URL = "https://api.themoviedb.org/3"

    if not API_KEY:
        return {"error": "TMDB API key is missing"}

    try:
        timeout = ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Search for the movie to get its ID
            search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={movie_name}"
            async with session.get(search_url) as resp:
                if resp.status != 200:
                    return {"error": "Failed to fetch movie info from TMDB."}
                data = await resp.json()
                if not data.get("results"):
                    return {"error": "Movie not found or no streaming info available."}
                movie_id = data["results"][0]["id"]

            # Fetch OTT providers using the movie ID
            provider_url = f"{BASE_URL}/movie/{movie_id}/watch/providers?api_key={API_KEY}"
            async with session.get(provider_url) as resp:
                if resp.status != 200:
                    return {"error": "Failed to fetch streaming info from TMDB."}
                providers_data = await resp.json()
                if "results" not in providers_data or not providers_data["results"]:
                    return {"error": "No streaming info available."}
                providers = providers_data["results"].get("IN", {}).get("flatrate", [])
                free_providers = providers_data["results"].get("IN", {}).get("free", [])
                return {
                    "Free": [p["provider_name"] for p in free_providers],
                    "Paid": [p["provider_name"] for p in providers]
                }
    except aiohttp.ClientError as e:
        logging.error(f"Network error in get_ott_links: {e}")
        return {"error": f"Network error: {e}"}
    except Exception as e:
        logging.error(f"Unexpected error in get_ott_links: {e}")
        return {"error": f"Unexpected error: {e}"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend_movie():
    movie_name = request.args.get("movie_name", "").strip()
    if not movie_name:
        return jsonify({"error": "Movie name is required!"}), 400

    recs = recommend(movie_name)
    if isinstance(recs, tuple):  # if error tuple is returned
        return jsonify(recs[0]), recs[1]
    return jsonify(recs)

def fetch_movies(api_url):
    """
    Utility function to fetch movies from a given API URL.
    """
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return jsonify(data.get('results', []))
        else:
            return jsonify({'error': 'Failed to fetch movies.'})
    except requests.RequestException as e:
        logging.error(f"Error in fetch_movies: {e}")
        return jsonify({'error': f'Request failed: {e}'})

def search_actor(actor_name):
    """
    Searches for an actor and then fetches movies featuring that actor.
    """
    API_KEY = os.getenv("TMDB_API_KEY")
    actor_api_url = f'https://api.themoviedb.org/3/search/person?api_key={API_KEY}&query={actor_name}'
    try:
        response = requests.get(actor_api_url)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                actor_id = data['results'][0]['id']
                actor_movies_url = f'https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_cast={actor_id}'
                return fetch_movies(actor_movies_url)
            else:
                return jsonify({'error': 'No actor found with this name.'})
        else:
            return jsonify({'error': 'Failed to fetch actor details.'})
    except requests.RequestException as e:
        logging.error(f"Error in search_actor: {e}")
        return jsonify({'error': f'Request failed: {e}'})

@app.route("/search", methods=["GET"])
def search_movies():
    search_query = request.args.get("query", "").strip()
    search_type = request.args.get("type", "").strip().lower()  # Optional: actor, genre, or movie
    
    if not search_query:
        return jsonify({"error": "Query is required!"}), 400

    # Actor search if explicitly specified
    if search_type == "actor":
        return search_actor(search_query)
    
    # Genre-based search if query matches known genres
    if search_query.lower() in genre_keywords:
        genre_id = genre_keywords[search_query.lower()]
        api_url = (f'https://api.themoviedb.org/3/discover/movie?api_key={os.getenv("TMDB_API_KEY")}'
                   f'&with_genres={genre_id}&sort_by=popularity.desc')
        return fetch_movies(api_url)
    
    # If query contains multiple words, try actor search first
    if len(search_query.split()) > 1:
        actor_result = search_actor(search_query)
        # If actor search returns an error, fallback to movie title search
        if actor_result.json.get("error"):
            pass
        else:
            return actor_result
    
    # Default: search by movie title
    api_url = f'https://api.themoviedb.org/3/search/movie?api_key={os.getenv("TMDB_API_KEY")}&query={search_query}'
    return fetch_movies(api_url)

@app.route("/ott", methods=["GET"])
def ott_route():
    movie_name = request.args.get("movie_name", "").strip()
    if not movie_name:
        return jsonify({"error": "Movie name is required!"}), 400
    ott_info = asyncio.run(get_ott_links(movie_name))
    return jsonify(ott_info)

@app.route("/movie_details", methods=["GET"])
def movie_details():
    if get_movie_full_details is None:
        return jsonify({"error": "get_movie_full_details function not available."}), 500

    movie_name = request.args.get("movie_name", "").strip()
    if not movie_name:
        return jsonify({"error": "Movie name is required!"}), 400

    details = asyncio.run(get_movie_full_details(movie_name))
    return jsonify(details)

@app.route("/full_recommendations", methods=["GET"])
def full_recommendations():
    if get_full_recommendations is None:
        return jsonify({"error": "get_full_recommendations function not available."}), 500

    movie_name = request.args.get("movie_name", "").strip()
    k = request.args.get("k", 19, type=int)
    if not movie_name:
        return jsonify({"error": "Movie name is required!"}), 400

    rec = asyncio.run(get_full_recommendations(movie_name, k))
    return jsonify(rec)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
