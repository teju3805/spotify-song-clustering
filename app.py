import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Spotify AI Music Cluster",
    page_icon="🎧",
    layout="wide"
)

# Load model and scaler
model = pickle.load(open("spotify_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Header
st.markdown("""
# 🎧 Spotify AI Music Cluster Predictor
Analyze song audio features and discover the **type of music cluster** using Machine Learning.
""")

st.divider()

# Sidebar controls
st.sidebar.header("🎛 Adjust Song Features")

danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.sidebar.slider("Loudness", -60.0, 0.0, -10.0)
tempo = st.sidebar.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎵 Input Audio Features")

    st.metric("Danceability", round(danceability,2))
    st.metric("Energy", round(energy,2))
    st.metric("Loudness", round(loudness,2))
    st.metric("Tempo", round(tempo,2))
    st.metric("Valence", round(valence,2))

with col2:
    st.subheader("📊 Feature Visualization")

    data = pd.DataFrame({
        "Feature": ["Danceability","Energy","Loudness","Tempo","Valence"],
        "Value": [danceability,energy,loudness,tempo,valence]
    })

    st.bar_chart(data.set_index("Feature"))

st.divider()

# Cluster descriptions
cluster_names = {
    0:"Chill / Acoustic Songs",
    1:"High Energy Dance Songs",
    2:"Emotional / Soft Songs",
    3:"Upbeat Party Songs",
    4:"Experimental / Mixed Style"
}

# Song recommendations
song_recommendations = {
    0:[
        "Someone Like You — Adele",
        "Let Her Go — Passenger",
        "Skinny Love — Bon Iver"
    ],

    1:[
        "Titanium — David Guetta",
        "Animals — Martin Garrix",
        "Don't You Worry Child — Swedish House Mafia"
    ],

    2:[
        "Stay With Me — Sam Smith",
        "All I Want — Kodaline",
        "Say You Won't Let Go — James Arthur"
    ],

    3:[
        "Levitating — Dua Lipa",
        "Blinding Lights — The Weeknd",
        "Uptown Funk — Bruno Mars"
    ],

    4:[
        "Bad Guy — Billie Eilish",
        "Starboy — The Weeknd",
        "Take Me To Church — Hozier"
    ]
}

# Prediction
if st.button("🔍 Predict Music Cluster"):

    features = np.array([[danceability, energy, loudness, tempo, valence]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    cluster = prediction[0]

    st.success(f"🎯 Predicted Cluster: {cluster}")

    st.info(f"🎶 Music Style: **{cluster_names.get(cluster)}**")

    st.subheader("🎵 Recommended Songs")

    for song in song_recommendations.get(cluster, []):
        st.write("•", song)

st.divider()

