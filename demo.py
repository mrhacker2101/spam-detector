# Importing necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Sample dataset (you can load a real dataset of Instagram comments)
# For this example, we'll create a small dataset. Replace with your actual dataset.
data = {
    'comment': [
        'Buy cheap sunglasses now!',
        'Hey, how are you?',
        'Click here for a free iPhone',
        'Join us for dinner tonight.',
        'Get followers instantly by visiting this link!',
        'I love your content!',
        'Free money! Just sign up!',
        'Looking forward to your next post!'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1: Spam, 0: Not Spam
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 1: Preprocess the text data
def preprocess_text(text):
    # Remove non-letter characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization (split text into words)
    text = text.split()
    
    # Stemming and removing stopwords
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = [ps.stem(word) for word in text if word not in stop_words]
    
    # Rejoin the words into a cleaned string
    return ' '.join(text)

# Apply the preprocessing function to the dataset
df['cleaned_comment'] = df['comment'].apply(preprocess_text)

# Step 2: Convert text into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['cleaned_comment']).toarray()
y = df['label']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a basic Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Use the model to predict spam in new comments
new_comments = [
    'You won a prize! Click here to claim it.',
    'Can’t wait for your next post!',
    'Get 1000 followers now, just sign up!'
]

# Preprocess the new comments
cleaned_new_comments = [preprocess_text(comment) for comment in new_comments]

# Convert to TF-IDF features
new_comments_tfidf = tfidf.transform(cleaned_new_comments).toarray()

# Make predictions
predictions = model.predict(new_comments_tfidf)

# Display predictions
for comment, pred in zip(new_comments, predictions):
    print(f"Comment: '{comment}' - Spam: {pred}")
