# Sentiment Analysis on Social Media

This project demonstrates how to perform sentiment analysis on social media texts using machine learning. The analysis is implemented using Python, Streamlit for the web interface, and scikit-learn for training a sentiment classifier.

## 📌 Project Overview

The goal of this project is to build an interactive web application that:
- Takes user input (social media post/text)
- Predicts the sentiment (e.g., Positive, Negative, Neutral)
- Visualizes the distribution of sentiments in the dataset

## Technologies Used

- **Python**
- **Streamlit** – for building the UI
- **Pandas, NumPy** – for data manipulation
- **Scikit-learn** – for building the sentiment classification model
- **Matplotlib, Seaborn** – for data visualization
- **Joblib** – for model serialization (in development)
- **Seaborn**-A Python library for statistical data visualization, can be used to visualize the results of sentiment analysis

## Dataset

The dataset (`sentimentdataset.csv`) contains two columns:
- `Text`: The actual content from social media.
- `Sentiment`: The sentiment label associated with the text (e.g., Positive, Negative).

Before training, the sentiment labels are stripped of leading/trailing whitespace to ensure label consistency.

