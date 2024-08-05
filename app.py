import pickle as pkl
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
        
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the pickle files 
model = pkl.load(open("Model.pkl",'rb'))

vector = pkl.load(open("tfidf_vectorizer.pkl",'rb'))

st.title("Email Spam classifier")

input_msg = st.text_input("Enter the mseeage")

if st.button('Predict'):
    transformed_msg = transform_text(input_msg)

    vectorizer_msg = vector.transform([transformed_msg])

    result = model.predict(vectorizer_msg)[0]

    if result==1:
        st.header("Spam!")
    else:
        st.header("Not Spam")





