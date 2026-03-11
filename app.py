import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Spotify AI Music Cluster Predictor",
    page_icon="🎧",
    layout="wide"
)

# Load ML model
model = pickle.load(open("spotify_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================= HEADER =================

st.title("🎧 Spotify AI Music Cluster Predictor")

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("""
### Student Details

**Name:** Your Name  
**Registration Number:** Your Registration Number  
**Class:** Your Class  

**Project Title:** Spotify Music Clustering using Machine Learning

🔗 **Google Colab Notebook:**  
https://colab.research.google.com/your-colab-link
""")

with col2:
    st.image("profile.jpg", width=200)

st.divider()

# ================= PROJECT OVERVIEW =================

st.header("📘 Project Overview")

st.write("""
Music streaming platforms contain millions of songs, making it difficult
for users to discover similar songs. This project uses **Machine Learning**
to automatically group songs with similar audio characteristics.

Using **K-Means Clustering**, songs are grouped based on their musical
features such as danceability, energy, loudness, tempo, and valence.

The model identifies patterns in music data and assigns songs into clusters
representing different music styles.
""")

# ================= PROBLEM STATEMENT =================

st.header("🎯 Problem Statement")

st.write("""
The objective of this project is to cluster songs based on their audio
features using unsupervised machine learning. By grouping similar songs,
the system can help discover hidden patterns in music and assist in
music recommendation systems.
""")

# ================= DATASET INFO =================

st.header("📊 Dataset Information")

st.write("""
Dataset Source: **Spotify Tracks Dataset (Kaggle)**

The dataset contains thousands of songs with multiple audio features
that describe the musical characteristics of each track.
""")

# ================= FEATURES =================

st.header("🎵 Features Used")

features_table = pd.DataFrame({
"Feature":[
"Danceability",
"Energy",
"Loudness",
"Tempo",
"Valence"
],

"Description":[
"Measures how suitable a track is for dancing",
"Represents intensity and activity level",
"Overall loudness of the track",
"Speed or pace of the music",
"Musical positivity or emotional tone"
]
})

st.table(features_table)

# ================= MODEL INFO =================

st.header("🤖 Machine Learning Method")

st.write("""
The model uses **K-Means Clustering**, an unsupervised learning algorithm.

Steps followed in the project:

1. Data preprocessing
2. Feature selection
3. Feature scaling using StandardScaler
4. Finding optimal clusters using Elbow Method
5. Training K-Means clustering model
6. Evaluating clusters using Silhouette Score
7. Deploying the model using Streamlit
""")

# ================= WORKFLOW =================

st.header("⚙ System Workflow")

st.markdown("""
Dataset → Data Preprocessing → Feature Scaling → K-Means Clustering  
→ Cluster Evaluation → Streamlit Deployment → Interactive Prediction
""")

st.divider()

# ================= PREDICTION TOOL =================

st.header("🎛 Interactive Music Cluster Predictor")

st.sidebar.header("Adjust Song Features")

danceability = st.sidebar.slider("Danceability",0.0,1.0,0.5)
energy = st.sidebar.slider("Energy",0.0,1.0,0.5)
loudness = st.sidebar.slider("Loudness",-60.0,0.0,-10.0)
tempo = st.sidebar.slider("Tempo",50.0,200.0,120.0)
valence = st.sidebar.slider("Valence",0.0,1.0,0.5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Feature Values")

    st.metric("Danceability",round(danceability,2))
    st.metric("Energy",round(energy,2))
    st.metric("Loudness",round(loudness,2))
    st.metric("Tempo",round(tempo,2))
    st.metric("Valence",round(valence,2))

with col2:
    st.subheader("Feature Visualization")

    chart_data = pd.DataFrame({
    "Feature":["Danceability","Energy","Loudness","Tempo","Valence"],
    "Value":[danceability,energy,loudness,tempo,valence]
    })

    st.bar_chart(chart_data.set_index("Feature"))

st.divider()

# ================= CLUSTER DETAILS =================

cluster_names = {
0:"Chill / Acoustic Songs",
1:"High Energy Dance Songs",
2:"Emotional / Soft Songs",
3:"Upbeat Party Songs",
4:"Experimental / Mixed Style"
}

song_recommendations = {

0:["Someone Like You — Adele",
"Let Her Go — Passenger",
"Skinny Love — Bon Iver"],

1:["Titanium — David Guetta",
"Animals — Martin Garrix",
"Don't You Worry Child — Swedish House Mafia"],

2:["Stay With Me — Sam Smith",
"All I Want — Kodaline",
"Say You Won't Let Go — James Arthur"],

3:["Levitating — Dua Lipa",
"Blinding Lights — The Weeknd",
"Uptown Funk — Bruno Mars"],

4:["Bad Guy — Billie Eilish",
"Starboy — The Weeknd",
"Take Me To Church — Hozier"]
}

# ================= PREDICTION =================

if st.button("Predict Music Cluster"):

    features = np.array([[danceability,energy,loudness,tempo,valence]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    cluster = prediction[0]

    st.success(f"Predicted Cluster: {cluster}")

    st.info(f"Music Style: {cluster_names.get(cluster)}")

    st.subheader("🎵 Recommended Songs")

    for song in song_recommendations.get(cluster,[]):
        st.write("•",song)

st.divider()

st.caption("Developed using Machine Learning and Streamlit")
