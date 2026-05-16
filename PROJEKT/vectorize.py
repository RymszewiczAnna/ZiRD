import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

print("Wczytywanie oczyszczonych danych...")
df = pd.read_parquet('movies_data_cleaned.parquet')

X = df['cleaned_review']
y = df['sentiment']


print("Dzielenie danych na zbiór treningowy i testowy...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tworzenie reprezentacji wektorowej 1: Bag of Words...")
bow_vectorizer = CountVectorizer(max_features=5000)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

print("Tworzenie reprezentacji wektorowej 2: TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nKształt danych BoW (treningowe): {X_train_bow.shape}")
print(f"Kształt danych TF-IDF (treningowe): {X_train_tfidf.shape}")

output_file = 'vectorized_data.pkl'
with open(output_file, 'wb') as f:

    pickle.dump({
        'X_train_bow': X_train_bow,
        'X_test_bow': X_test_bow,
        'X_train_tfidf': X_train_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'y_train': y_train,
        'y_test': y_test
    }, f)

print(f"\nGotowe macierze liczbowe oraz etykiety zapisano do pliku: {output_file}")