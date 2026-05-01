
# Detecting Inauthentic Profiles in Dating Apps using hybrid gcn-gru

## 🧠 Overview

Online dating platforms face a growing challenge: fake profiles, bots, and coordinated spam behavior.

This project builds a production-style deep learning system that detects such inauthentic profiles by combining:

🔗 Graph Intelligence (GCN) → Learns relationships between users

⏳ Behavior Modeling (GRU) → Captures evolving activity patterns

Unlike traditional ML models, this approach analyzes users in context, not isolation.

## 🧪 Key Features
🧠 Hybrid Deep Learning Model (GCN + GRU)

🌐 Graph-Based User Similarity Network

⏱️ Sequential Behavioral Pattern Learning

⚖️ Multi-Class Risk Classification

📊 Interactive Streamlit Dashboard

📄 Automated Risk Report Generation (PDF)

## 🏗️ System Architecture
🔹 1. Data Processing
Feature engineering from user activity

Rule-based anomaly scoring for pseudo-labeling

Normalization for stable training

🔹 2. Graph Construction

Users connected using cosine similarity

Top-K nearest neighbors form edges

Captures hidden behavioral clusters

🔹 3. Hybrid Model

GCN (Graph Convolutional Network)

→ Learns structural embeddings from the user graph

GRU (Gated Recurrent Unit)

→ Learns behavior patterns from user activity sequences

Fusion Layer

→ Combines both embeddings for final classification

🔹 4. Output Classes

✅ Authentic

⚠️ Potentially Inauthentic

🚨 Inauthentic

## 📊 Interactive Dashboard
🎯 Risk Score (0–100)

📉 Behavioral comparison vs network average

🌐 3D user connection graph

📊 Confidence distribution charts

📄 Downloadable PDF risk reports

## 🛠️ Tech Stack

PyTorch

PyTorch Geometric

Streamlit

Plotly

NetworkX

Scikit-learn

Pandas / NumPy

## ▶️ How to Run

# 1. Train the Model
    python train.py
## 2. Evaluate Performance
    python evaluate.py
## 3. Launch Dashboard
    streamlit run app.py

📈 Evaluation Metrics

Accuracy

Confusion Matrix

Classification Report

## 🔮 Future Improvements

Incorporate real-time streaming data

Use Graph Attention Networks (GAT)

Add explainability (SHAP / attention visualization)

Deploy as a scalable API service

# 🖼️Screenshots


<img width="1888" height="967" alt="Screenshot 2026-04-29 114827" src="https://github.com/user-attachments/assets/9be52a8a-0438-4aea-9f9a-5d632920bee0" />


A webpage with a dashboard to search and filter the profiles 


<img width="1366" height="808" alt="Screenshot 2026-04-29 114955" src="https://github.com/user-attachments/assets/c6268d1d-e2e4-4ed9-aafc-542777a9a4f0" />

Graph constructed using cosine similarity and visualised using kNN with 15 neighbouring profiles




<img width="1394" height="546" alt="Screenshot 2026-04-29 115024" src="https://github.com/user-attachments/assets/a8d8bf8d-46c1-4bf8-9cf8-46a950ba2358" />



Fused features from GCN-GRU and passed on to the softmax function to produce a probability score.
The output has been visualized in the risk meter, and the confidence score is displayed



<img width="1438" height="539" alt="Screenshot 2026-04-29 115047" src="https://github.com/user-attachments/assets/366c7698-1d2a-4b1d-8b3c-5a579e26060e" />

Behavioral analysis of the profile on the given metrics and visualized in a radar chart


