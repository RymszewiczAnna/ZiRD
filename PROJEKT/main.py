import pandas as pd
from bs4 import BeautifulSoup
import os
import cloudscraper



def load_csv_data(filepath='IMDB Dataset.csv'):
    """
    Wczytuje recenzje z gotowego pliku CSV.
    Domyślnie szuka pliku 'IMDB Dataset.csv' z Kaggle.
    """
    if not os.path.exists(filepath):
        print(f"BŁĄD: Nie znaleziono pliku '{filepath}'.")
        print("Pobierz 'IMDB Dataset of 50K Movie Reviews' z Kaggle,")
        print("wypakuj go i umieść w tym samym folderze co ten skrypt.")
        return pd.DataFrame(columns=['review', 'sentiment'])

    print(f"Wczytywanie danych z pliku: {filepath}...")
    df = pd.read_csv(filepath)


    df.columns = [col.lower() for col in df.columns]
    print(f"Pomyślnie wczytano {len(df)} recenzji z CSV.")

    return df



def scrape_letterboxd_reviews(movie_url):
    """
    Pobiera recenzje użytkowników z podanego adresu URL na Letterboxd,
    wykorzystując cloudscraper do ominięcia błędu 403.
    """
    print(f"Uruchamianie scrapera dla adresu: {movie_url}...")

    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )

    try:

        response = scraper.get(movie_url, timeout=15)

        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        review_containers = soup.find_all('div', class_='body-text')

        scraped_data = []
        for review in review_containers:
            tekst_recenzji = review.get_text(separator=' ', strip=True)

            if len(tekst_recenzji) > 10:
                scraped_data.append({
                    'review': tekst_recenzji,
                    'sentiment': 'scraped_unlabeled'
                })

        df_scraped = pd.DataFrame(scraped_data)
        print(f"Scraper pobrał pomyślnie {len(df_scraped)} recenzji z tej strony.")
        return df_scraped

    except Exception as e:
        print(f"Błąd podczas łączenia ze stroną: {e}")
        return pd.DataFrame(columns=['review', 'sentiment'])

def process_and_save_data(df_csv, df_scraped, output_filename='movies_data.parquet'):
    """Łączy oba źródła, czyści puste wiersze i zapisuje do formatu Parquet."""
    print("\nŁączenie i walidacja danych...")

    combined_df = pd.concat([df_csv, df_scraped], ignore_index=True)

    initial_shape = combined_df.shape[0]

    combined_df.dropna(subset=['review'], inplace=True)

    combined_df.drop_duplicates(subset=['review'], inplace=True)

    final_shape = combined_df.shape[0]
    print(f"Walidacja zakończona. Usunięto {initial_shape - final_shape} błędnych/pustych/zduplikowanych wierszy.")

    combined_df.to_parquet(output_filename)
    print(f"Sukces! Zapisano {final_shape} recenzji do ostatecznego pliku: {output_filename}")

    return combined_df


if __name__ == "__main__":

    nazwa_pliku_csv = 'IMDB Dataset.csv'
    df_kaggle = load_csv_data(nazwa_pliku_csv)

    url_do_scrapowania = "https://letterboxd.com/film/dune-part-two/reviews/"
    df_internet = scrape_letterboxd_reviews(url_do_scrapowania)

    if not df_kaggle.empty or not df_internet.empty:
        final_dataset = process_and_save_data(df_kaggle, df_internet)

        print("\nPróbka połączonych danych (5 pierwszych wierszy):")
        print(final_dataset.head())
        print("\nPróbka danych ze scrapera umieszczonych na końcu:")
        print(final_dataset.tail())
    else:
        print("\nBrak danych do zapisania.")