# Menginstall packages yang dibutuhkan
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Mengubah narasi menjadi bentuk yang telah di-stem dan tidak mengandung stopword
# Mengambil dua kata yang sama namun memiliki arti yang berbeda secara acak untuk setiap narasi
# Inisialisasi stopword remover dan stemmer
import random
stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_remover = stopword_factory.create_stop_word_remover()

# Configuration
data_train = "Detect/data_train/data_train.csv"


# Mengambil dua kata yang sama namun mengandung beda arti secara acak
def process_narasi(narasi):
    # Stemming
    stemmed_narasi = stemmer.stem(narasi)

    # Menghapus stopword
    clean_narasi = stopword_remover.remove(stemmed_narasi)

    # Membagi narasi menjadi kata-kata
    words = clean_narasi.split()

    # Memilih dua kata yang berbeda secara acak
    selected_words = []

    if len(words) > 1:
        # Memilih dua kata secara acak
        selected_words = random.sample(words, 2)
    elif len(words) == 1:
        # Jika hanya terdapat satu kata, duplikat kata tersebut
        selected_words = [words[0], words[0]]
    else:
        # Jika tidak ada kata dalam narasi, berikan nilai kosong
        selected_words = ["", ""]

    return ' '.join(selected_words)


def train_model():
    datasets = pd.read_csv(data_train)
    print(datasets["label"].value_counts())

    # Membagi dataset menjadi subset data latih (train) dan data uji (test) yang akan digunakan dalam pelatihan dan evaluasi model
    X_train, X_test, y_train, y_test = train_test_split(datasets['narasi'], datasets['label'], test_size=0.2, random_state=42)

    # Membentuk model Word2Vec berdasarkan teks yang terdapat dalam X_train
    sentences = [text.split() for text in X_train]
    model = Word2Vec(sentences, min_count=1)

    # Menghasilkan vektor representasi dari setiap kalimat dalam X_train dan X_test menggunakan model Word2Vec yang telah dibentuk sebelumnya.
    X_train_word_vectors = np.array(
        [
            np.mean(
                [model.wv[word] for word in sentence.split() if word in model.wv],
                axis=0
            )
            if any(word in model.wv for word in sentence.split())
            else np.zeros(model.vector_size)
            for sentence in X_train
        ]
    )

    X_test_word_vectors = np.array(
        [
            np.mean(
                [model.wv[word] for word in sentence.split() if word in model.wv],
                axis=0
            )
            if any(word in model.wv for word in sentence.split())
            else np.zeros(model.vector_size)
            for sentence in X_test
        ]
    )
    # Memastikan bahwa jumlah fitur (jumlah dimensi) dalam X_train_word_vectors dan X_test_word_vectors adalah sama sebelum melanjutkan proses selanjutnya.
    if X_train_word_vectors.shape[1] == X_test_word_vectors.shape[1]:
        X_train_word_vectors = X_train_word_vectors.astype(np.float32)
        X_test_word_vectors = X_test_word_vectors.astype(np.float32)

    else:
        print("Jumlah fitur pada X_train_word_vectors dan X_test_word_vectors berbeda.")

    return X_train_word_vectors, X_test_word_vectors, y_train, y_test


def predict(text):
    classifier = RandomForestClassifier()
    classifier.fit(X_train_word_vectors, y_train)
    predicted = classifier.predict(X_test_word_vectors)

    if predicted[0] == 1:
        return "Berita tersebut kemungkinan Palsu"
    else:
        return "Berita tersebut kemungkinan Asli"


# Train Machine Learning model
X_train_word_vectors, X_test_word_vectors, y_train, y_test = train_model()

# Streamlit app

st.title("Deteksi Berita")
st.write("Aplikasi ini digunakan untuk melakukan prediksi berita asli atau palsu. Namun aplikasi ini tidak dapat dijadikan acuan.")

input_text = st.text_area("Masukkan teks berita")

if st.button("Prediksi"):
    if input_text.strip() != "":
        prediction = predict(input_text)
        st.success(f"Hasil Prediksi: {prediction}")
    else:
        st.warning("Masukkan teks berita untuk melakukan prediksi.")

