 
import pickle
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load models
movies = pickle.load(open("models/movies.pkl", "rb"))
vectorizer = joblib.load("models/vectorizer.joblib")

knn_model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
knn_model.fit(vectorizer)

joblib.dump(knn_model, "models/knn_model.joblib")
