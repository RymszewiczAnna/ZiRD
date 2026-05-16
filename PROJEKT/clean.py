import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')



def clean_and_normalize_text(text):
    """
    Funkcja przyjmuje surowy tekst recenzji i przeprowadza
    pełen proces czyszczenia i normalizacji NLP.
    """

    text = BeautifulSoup(text, "html.parser").get_text()

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

if __name__ == "__main__":
    print("Wczytywanie zapisanych danych...")
    df = pd.read_parquet('movies_data.parquet')

    print("Rozpoczynam przetwarzanie tekstu NLP")

    df['cleaned_review'] = df['review'].apply(clean_and_normalize_text)

    output_file = 'movies_data_cleaned.parquet'
    df.to_parquet(output_file)

    print("\nOto porównanie tekstu przed i po czyszczeniu:")

    print(f"SUROWY:\n{df.iloc[0]['review']}")
    print(f"CZYSTY:\n{df.iloc[0]['cleaned_review']}")

    print(f"\nSukces! Oczyszczone dane zapisano jako {output_file}")