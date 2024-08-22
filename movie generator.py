import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample Data (Replace this with your dataset)
data = {
    'plot': [
        "A young boy discovers he is a wizard on his 11th birthday.",
        "A group of astronauts embark on a mission to save Earth from a catastrophic event.",
        "A romantic story set in a small town during the 1940s.",
        "A detective must solve a complex murder case in a futuristic city.",
        "A group of friends go on a road trip across the country."
    ],
    'genre': ['Fantasy', 'Sci-Fi', 'Romance', 'Thriller', 'Comedy']
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['plot'], df['genre'], test_size=0.2, random_state=42)

# Vectorizing text data with TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize classifiers
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(kernel='linear')
}

# Train, predict, and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 60)
