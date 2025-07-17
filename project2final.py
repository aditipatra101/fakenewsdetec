import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv('data/fake_or_real_news.csv')

# Clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    return ' '.join([word for word in text.split() if word not in stop_words])

df['cleaned'] = df['text'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_df=0.7)
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
