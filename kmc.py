import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# App title
st.title("🔵 K-Means Clustering Demo")

# Sample dataset (Customer Segmentation)
data = {
    "Annual_Income": [15, 16, 17, 18, 20, 25, 30, 35, 40, 45, 50, 60],
    "Spending_Score": [39, 42, 35, 30, 50, 60, 65, 70, 80, 85, 90, 95]
}

df = pd.DataFrame(data)

st.subheader("📊 Dataset")
st.write(df)

# Select number of clusters
k = st.slider("Select number of clusters (K)", min_value=2, max_value=6, value=3)

# Feature selection
X = df[["Annual_Income", "Spending_Score"]]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans model
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)

st.subheader("📌 Clustered Data")
st.write(df)

# User input
st.subheader("🔮 Predict Customer Cluster")
income = st.number_input("Enter annual income:")
spending = st.number_input("Enter spending score:")

if st.button("Predict Cluster"):
    new_data = scaler.transform([[income, spending]])
    cluster = model.predict(new_data)
    st.success(f"Customer belongs to Cluster {cluster[0]}")
