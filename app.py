import string

import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def preprocess_word(word):
    # -----------Lower Case---------------#
    word = word.lower()
    # -----------Tokenization-------------#
    word = nltk.word_tokenize(word)

    # ----------Removing Special Characters (@#$!)---------------#
    abc = []
    for i in word:
        if i.isalnum():
            abc.append(i)

    # -----------Removing Stop Words and Punctuation---------------#
    # Stop Words - {i,me,myself,our,we,our,you,yours}
    # Punctuations - {.,""\[]()}

    word = abc[:]
    abc.clear()

    for i in word:
        if i not in stopwords.words('english') and i not in string.punctuation:
            abc.append(i)

    word = abc[:]
    abc.clear()

    for i in word:
        abc.append(ps.stem(i))

    return " ".join(abc)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email Spam Classification')

input_sms = st.text_area('Enter The Message')
if st.button('Predict'):

    transform_sms = preprocess_word(input_sms)

    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header('its_Spam')
    else:
        st.header('its_Ham')
