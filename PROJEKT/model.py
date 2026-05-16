import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Wczytywanie przygotowanych danych wektorowych...")
with open('vectorized_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train_tfidf']
X_test = data['X_test_tfidf']
y_train = data['y_train']
y_test = data['y_test']

mask_train = y_train != 'scraped_unlabeled'
X_train_labeled = X_train[mask_train.values]
y_train_labeled = y_train[mask_train]

mask_test = y_test != 'scraped_unlabeled'
X_test_labeled = X_test[mask_test.values]
y_test_labeled = y_test[mask_test]

mask_scraped = y_test == 'scraped_unlabeled'
X_scraped = X_test[mask_scraped.values]

print(f"Ilość recenzji do treningu: {X_train_labeled.shape[0]}")
print(f"Ilość recenzji do testowania: {X_test_labeled.shape[0]}")

print("\nRozpoczynam trenowanie modelu Random Forest.")
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train_labeled, y_train_labeled)
print("Trening zakończony sukcesem!")

print("\n--- WYNIKI MODELU NA ZBIORZE TESTOWYM ---")

y_pred = model.predict(X_test_labeled)

dokladnosc = accuracy_score(y_test_labeled, y_pred)
print(f"Dokładność (Accuracy): {dokladnosc * 100:.2f}%\n")

print("Raport Klasyfikacji:")
print(classification_report(y_test_labeled, y_pred))

