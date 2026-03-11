import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("spotify_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Spotify Song Cluster Predictor 🎵")

st.write("Enter song audio features to predict cluster.")

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

if st.button("Predict Cluster"):

    features = np.array([[danceability, energy, loudness, tempo, valence]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    st.success(f"Predicted Cluster: {prediction[0]}")
