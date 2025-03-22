import os
import ast
import logging
import asyncio
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import load, dump
import aiohttp
from aiohttp import ClientTimeout
import nest_asyncio
import requests  # For TMDB API fallback
import pandas as pd

# Setup logging with a clear format
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Apply fix for Jupyter Notebook compatibility
nest_asyncio.apply()

# Get TMDB API key from environment variables
TMDB_API_KEY: str = os.environ.get("TMDB_API_KEY", "3075ab4eebb27e51a67f7869d6b7984a")

# Try loading the pre-trained ML model and vectorizer from the 'models' folder
try:
    knn_model = load(os.path.join("models", "knn_model.joblib"))
    vectorizer = load(os.path.join("models", "vectorizer.joblib"))
    movies_df = pd.read_csv(os.path.join("models", "movies_data.csv"))
    logger.info("Pre-trained ML model, vectorizer, and movies mapping loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ML model: {e}")
    knn_model, vectorizer, movies_df = None, None, None

def recommend_tmdb(movie_name: str, k: int = 5) -> Union[Dict[str, Any], Dict[str, str]]:
    """Return recommended movies using TMDB API fallback."""
    movie_name_proc = movie_name.lower().strip()
    if not TMDB_API_KEY:
        return {"error": "TMDB API key is missing."}
    
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name_proc}"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code != 200:
            return {"error": "TMDB API search failed."}
        data = response.json()
        if not data.get("results"):
            return {"error": "Movie not found in TMDB."}
        movie_id = data["results"][0]["id"]
        similar_url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={TMDB_API_KEY}&language=en-US&page=1"
        similar_resp = requests.get(similar_url, timeout=10)
        if similar_resp.status_code != 200:
            return {"error": "TMDB API similar movies fetch failed."}
        similar_data = similar_resp.json()
        if not similar_data.get("results"):
            return {"error": "No similar movies found on TMDB."}
        recommended = [movie["title"] for movie in similar_data["results"]][:k]
        return {"recommended_movies": recommended, "source": "TMDB fallback"}
    except Exception as e:
        return {"error": f"Error during TMDB fallback: {e}"}

def recommend_ml(movie_name: str, k: int = 5) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Return recommended movies using the pre-trained ML model.
    It uses the vectorizer to transform the input movie's text (e.g., tags or title) 
    and then finds nearest neighbors from movies_df.
    """
    if knn_model is None or vectorizer is None or movies_df is None:
        return {"error": "Pre-trained ML model or movies mapping is not available."}
    
    # Assume your model was trained on a 'tags' column; here we use the movie title as input
    movie_name_proc = movie_name.lower().strip()
    input_vector = vectorizer.transform([movie_name_proc])
    
    try:
        distances, indices = knn_model.kneighbors(input_vector, n_neighbors=k+1)
        # Exclude the input movie itself (assumed to be the first neighbor)
        rec_indices = indices.squeeze().tolist()[1:]
        recommended_titles = movies_df.iloc[rec_indices]['title'].tolist()
        return {"recommended_movies": recommended_titles, "source": "ML model"}
    except Exception as e:
        return {"error": f"ML recommendation failed: {e}"}

async def get_movie_full_details(movie_name: str) -> Dict[str, Any]:
    BASE_URL = "https://api.themoviedb.org/3"
    try:
        timeout = ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            search_url = f"{BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
            async with session.get(search_url) as resp:
                if resp.status != 200:
                    return {"error": "Failed to fetch movie info from TMDB."}
                data = await resp.json()
                if not data.get("results"):
                    return {"error": "Movie not found."}
                movie_data = data["results"][0]
                movie_id = movie_data["id"]
                poster_url = (f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}"
                              if movie_data.get("poster_path") else None)

            details_url = f"{BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
            async with session.get(details_url) as resp:
                if resp.status != 200:
                    return {"error": "Failed to fetch additional movie details."}
                movie_details = await resp.json()

            credits_url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
            async with session.get(credits_url) as resp:
                if resp.status != 200:
                    cast, crew = [], []
                else:
                    credits = await resp.json()
                    cast = [member["name"] for member in credits.get("cast", [])[:3]]
                    crew = [member["name"] for member in credits.get("crew", []) if member.get("job") == "Director"]

            return {
                "title": movie_details.get("title"),
                "overview": movie_details.get("overview"),
                "rating": movie_details.get("vote_average"),
                "release_date": movie_details.get("release_date"),
                "cast": ", ".join(cast),
                "crew": ", ".join(crew),
                "poster_url": poster_url
            }
    except aiohttp.ClientError as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

async def get_ott_links(movie_name: str) -> Dict[str, Any]:
    BASE_URL = "https://api.themoviedb.org/3"
    try:
        timeout = ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            search_url = f"{BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
            async with session.get(search_url) as resp:
                if resp.status != 200:
                    return {"error": "Failed to fetch movie info from TMDB."}
                data = await resp.json()
                if not data.get("results"):
                    return {"error": "No streaming info available."}
                movie_id = data["results"][0]["id"]
            provider_url = f"{BASE_URL}/movie/{movie_id}/watch/providers?api_key={TMDB_API_KEY}"
            async with session.get(provider_url) as resp:
                if resp.status != 200:
                    return {"error": "Failed to fetch streaming info from TMDB."}
                providers_data = await resp.json()
                results = providers_data.get("results", {})
                in_country = results.get("IN", {})
                providers = in_country.get("flatrate", [])
                free_providers = in_country.get("free", [])
                return {
                    "Free": [p.get("provider_name") for p in free_providers],
                    "Paid": [p.get("provider_name") for p in providers]
                }
    except aiohttp.ClientError as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

async def get_full_recommendations(movie_name: str, k: int = 5) -> Dict[str, Any]:
    # First, try the ML-based recommendation.
    rec_result = recommend_ml(movie_name, k)
    if "error" in rec_result:
        logger.info("Falling back to TMDB API for recommendations.")
        rec_result = recommend_tmdb(movie_name, k)
    
    recommended_titles = rec_result.get("recommended_movies", [])
    tasks = []
    for title in recommended_titles:
        tasks.append(asyncio.gather(
            get_movie_full_details(title),
            get_ott_links(title)
        ))
    results = await asyncio.gather(*tasks)
    full_recommendations = []
    for (details, ott) in results:
        movie_info = {
            "title": details.get("title", "N/A"),
            "overview": details.get("overview", "N/A"),
            "rating": details.get("rating", "N/A"),
            "release_date": details.get("release_date", "N/A"),
            "cast": details.get("cast", "N/A"),
            "crew": details.get("crew", "N/A"),
            "poster_url": details.get("poster_url"),
            "ott_availability": ott if "error" not in ott else "Not Available"
        }
        full_recommendations.append(movie_info)
    return {"recommended_movies": full_recommendations}

# For testing purposes: Run a series of tests when this module is executed directly.
if __name__ == "__main__":
    test_movie = "Avatar"
    
    # Test ML-based recommendation first
    ml_rec = recommend_ml(test_movie)
    logger.info(f"ML-based recommendations for '{test_movie}': {ml_rec}")
    
    # Test TMDB fallback (if needed)
    tmdb_rec = recommend_tmdb(test_movie)
    logger.info(f"TMDB fallback recommendations for '{test_movie}': {tmdb_rec}")
    
    # Test OTT links for base movie
    ott_result = asyncio.run(get_ott_links(test_movie))
    logger.info(f"OTT availability for '{test_movie}': {ott_result}")
    
    # Test full movie details for base movie
    details_result = asyncio.run(get_movie_full_details(test_movie))
    logger.info(f"Full details for '{test_movie}': {details_result}")
    
    # Test full recommendations with detailed info
    full_rec_result = asyncio.run(get_full_recommendations(test_movie))
    logger.info(f"Full recommendations for '{test_movie}': {full_rec_result}")
