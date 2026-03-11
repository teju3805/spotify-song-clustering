import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="Spotify AI Music Cluster",
    page_icon="🎧",
    layout="wide"
)

# Load model
model = pickle.load(open("spotify_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Header
st.markdown(
"""
# 🎧 Spotify AI Music Cluster Predictor
Discover what **type of song** your audio features represent using **Machine Learning clustering**.
""")

st.divider()

# Sidebar controls
st.sidebar.header("🎛 Adjust Song Features")

danceability = st.sidebar.slider("Danceability",0.0,1.0,0.5)
energy = st.sidebar.slider("Energy",0.0,1.0,0.5)
loudness = st.sidebar.slider("Loudness",-60.0,0.0,-10.0)
tempo = st.sidebar.slider("Tempo",50.0,200.0,120.0)
valence = st.sidebar.slider("Valence",0.0,1.0,0.5)

# Layout columns
col1, col2 = st.columns([1,1])

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
        "Feature":["Danceability","Energy","Loudness","Tempo","Valence"],
        "Value":[danceability,energy,loudness,tempo,valence]
    })

    st.bar_chart(data.set_index("Feature"))

st.divider()

# Predict button
if st.button("🔍 Predict Music Cluster"):

    features = np.array([[danceability,energy,loudness,tempo,valence]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    cluster = prediction[0]

    cluster_names = {
        0:"Chill / Acoustic Songs",
        1:"High Energy Dance Songs",
        2:"Emotional / Soft Songs",
        3:"Upbeat Party Songs",
        4:"Experimental / Mixed Style"
    }

    st.success(f"🎯 Predicted Cluster: {cluster}")

    st.info(f"🎶 Music Style: **{cluster_names.get(cluster)}**")

st.divider()

st.caption("Built with ❤️ using Machine Learning + Streamlit")
