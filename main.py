# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
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


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")


input_sms= st.text_area("Enter your message")

if st.button("Predict"):
    transformed_sms=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)

    if result==0:
        st.header("Not Spam")
    else :
        st.header("Spam")




# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
