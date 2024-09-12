Airline Sentiment Analysis Using Naive Bayes and Logistic Regression
This project involves analyzing and classifying airline-related tweets to determine their sentiment (positive, negative, neutral). The main steps include text preprocessing, feature extraction using TF-IDF, and training machine learning models to classify the sentiment of tweets.

Project Overview
The goal of this project is to:

Preprocess text data (airline tweets)
Use TF-IDF for feature extraction
Train machine learning models (Naive Bayes and Logistic Regression) to classify the sentiment of the tweets
Visualize key findings using word clouds
Dataset
The dataset used in this project contains tweets related to airlines and their customer service. It includes the following columns:

text: The tweet content
airline_sentiment: The sentiment label (positive, neutral, or negative)
Steps in the Analysis
Data Preprocessing:

Convert text to lowercase
Remove non-alphanumeric characters
Remove stopwords
Apply stemming and lemmatization
Feature Extraction:

Use TF-IDF (Term Frequency-Inverse Document Frequency) to transform text data into numeric features for model training
Model Training:

Train and evaluate two models:
Naive Bayes Classifier
Logistic Regression
Visualization:

Word clouds are generated for original, preprocessed, and TF-IDF-transformed text data.
Installation
Prerequisites
Python 3.x
Required Libraries:
pandas
nltk
sklearn
matplotlib
wordcloud
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/airline-sentiment-analysis.git
cd airline-sentiment-analysis
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset and place it in the project directory. Make sure the dataset is named Tweets.csv.

Running the Project
To preprocess the text, train the models, and evaluate their performance, run the Python script:

bash
Copy code
python sentiment_analysis.py
Output
Naive Bayes Model:
Accuracy, classification report, and confusion matrix for the test set
Logistic Regression Model:
Accuracy, classification report, and confusion matrix for the test set
Visualizations:
Word clouds showing frequent terms in the original, preprocessed, and TF-IDF-transformed text data
License
This project is licensed under the MIT License.
