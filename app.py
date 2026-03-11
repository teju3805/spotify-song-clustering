import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page settings
st.set_page_config(page_title="Spotify Song Cluster Predictor", page_icon="🎵")

# Load model
model = pickle.load(open("spotify_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎧 Spotify Song Cluster Predictor")
st.write("Enter song audio features to discover its music cluster.")

# Sliders
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

if st.button("Predict Cluster"):

    features = np.array([[danceability, energy, loudness, tempo, valence]])
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)
    cluster = prediction[0]

    cluster_names = {
        0: "Chill / Acoustic Songs",
        1: "High Energy Dance Songs",
        2: "Emotional / Soft Songs",
        3: "Upbeat Party Songs",
        4: "Experimental / Mixed Style"
    }

    st.success(f"Cluster: {cluster}")
    st.write("Song Type:", cluster_names.get(cluster))

    # Visualization
    st.subheader("Feature Visualization")

    data = pd.DataFrame({
        "Feature": ["Danceability", "Energy", "Loudness", "Tempo", "Valence"],
        "Value": [danceability, energy, loudness, tempo, valence]
    })

    st.bar_chart(data.set_index("Feature"))
