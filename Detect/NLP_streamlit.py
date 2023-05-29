import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk import ngrams
import streamlit as st

# Random seed for consistency
np.random.seed(500)

# Uncomment line below to download nltk requirements
# nltk.download('punkt')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
Encoder = LabelEncoder()
Tfidf_vect = TfidfVectorizer()

# Configuration
DATA_LATIH = "./Detect/Data Latih/DATA_TRAIN.csv"
DATA_UJI = "./Detect/Data Uji/Data Uji BDC.csv"


def preprocess_text(text):
    lower = stemmer.stem(text.lower())
    tokens = word_tokenize(lower)
    n = 2  # Panjang n-gram yang diinginkan
    ngram_lower = list(ngrams(tokens, n))
    ngram_text = " ".join([" ".join(ngram) for ngram in ngram_lower])
    return ngram_text


def train_model():
    datasets = pd.read_csv(DATA_LATIH)
    print(datasets["label"].value_counts())

    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]
    n = 2  # Panjang n-gram yang diinginkan
    ngram_lower = [list(ngrams(tokens, n)) for tokens in vectors]
    ngram_texts = [" ".join([" ".join(ngram) for ngram in text]) for text in ngram_lower]
    labels = datasets["label"]

    X_train, X_test, y_train, y_test = train_test_split(ngram_texts, labels, test_size=0.3, stratify=labels)

    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)

    Tfidf_vect.fit(ngram_texts)

    Train_X_Tfidf = Tfidf_vect.transform(X_train)
    Test_X_Tfidf = Tfidf_vect.transform(X_test)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=1, gamma="auto", verbose=True)
    SVM.fit(Train_X_Tfidf, y_train)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test)*100)
    return SVM


def predict(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = Tfidf_vect.transform([preprocessed_text])
    prediction = SVM.predict(text_tfidf)
    return prediction[0]


def test_model(SVM):
    datasets = pd.read_csv(DATA_UJI)

    lower = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]
    n = 2  # Panjang n-gram yang diinginkan
    ngram_lower = [list(ngrams(tokens, n)) for tokens in vectors]
    ngram_texts = [" ".join([" ".join(ngram) for ngram in text]) for text in ngram_lower]

    Test_X_Tfidf = Tfidf_vect.transform(ngram_texts)

    predictions_SVM = SVM.predict(Test_X_Tfidf)

    data = {"ID": list(datasets["ID"]), "prediksi": predictions_SVM}
    hasil = pd.DataFrame(data, columns=["ID", "prediksi"])
    hasil.to_csv("./Hasil Uji Model.csv", index=False)


# Train Machine Learning model
SVM = train_model()

# Streamlit app
st.title("Deteksi Sentimen Berita")
st.write("Aplikasi ini digunakan untuk melakukan prediksi sentimen pada teks berita.")

input_text = st.text_area("Masukkan teks berita")

if st.button("Prediksi"):
    if input_text.strip() != "":
        prediction = predict(input_text)
        sentiment = Encoder.inverse_transform([prediction])[0]
        st.success(f"Hasil Prediksi: {sentiment}")
    else:
        st.warning("Masukkan teks berita untuk melakukan prediksi.")
