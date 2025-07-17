
import streamlit as st
import joblib
import string

st.title("📰 Fake News Detection System")

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_input(text):
    return ' '.join([word for word in text.lower().split() if word not in string.punctuation])

user_input = st.text_area("Paste news content here:")

if st.button("Check if Fake"):
    cleaned = clean_input(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    st.write("### Result:", "🟥 FAKE" if prediction == "FAKE" else "🟩 REAL")
