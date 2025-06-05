import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("sentimentdataset.csv")
df['Sentiment'] = df['Sentiment'].str.strip()

# Train model
X = df['Text']
y = df['Sentiment']
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X, y)

# UI
st.title("Social Media Sentiment Analysis")

user_input = st.text_area("Enter your message:")
if st.button("Analyze"):
    prediction = pipeline.predict([user_input])[0]
    st.write(f"### Sentiment: {prediction}")

# Visualization
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Sentiment', ax=ax)
st.pyplot(fig)
