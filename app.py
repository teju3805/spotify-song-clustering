import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page configuration
st.set_page_config(page_title="Spotify AI Music Cluster", page_icon="🎧", layout="wide")

# Load model
model = pickle.load(open("spotify_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🎧 Spotify AI Music Cluster Predictor")
st.write("Analyze song audio features and discover the type of music cluster.")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎵 Song Audio Features")

    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
    tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)

with col2:
    st.subheader("📊 Feature Visualization")

    data = pd.DataFrame({
        "Feature": ["Danceability", "Energy", "Loudness", "Tempo", "Valence"],
        "Value": [danceability, energy, loudness, tempo, valence]
    })

    st.bar_chart(data.set_index("Feature"))

# Prediction
if st.button("Predict Music Cluster"):

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

    st.success(f"Predicted Cluster: {cluster}")
    st.info(f"Music Style: {cluster_names.get(cluster)}")
