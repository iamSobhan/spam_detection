import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


tfidf = pickle.load(open("tfidf.pkl", "rb"))
mnb_model = pickle.load(open("mnb_model.pkl", "rb"))

def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Spam Detection using Machine Learning")

input_sms = st.text_area("Enter the Text")


if st.button("Let's Predict"):

    transformed_sms = transform_message(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = mnb_model.predict(vector_input)[0]

    if result == 1:
        st.header("This is a Spam text.")
    else:
        st.header("This is not a Spam text.")