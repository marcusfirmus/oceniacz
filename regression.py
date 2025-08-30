#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sys
import os
import argparse
import numpy as np

# --- Konfiguracja modeli regresji ---
# Tutaj możesz wybrać, którego regresora użyć.
# Odkomentuj ten, którego chcesz użyć, i zakomentuj pozostałe.

# Dla przewidywania czasu wykonania:
# current_time_regressor = LinearRegression() # Prostota, dobra baza
# current_time_regressor = Ridge(alpha=1.0) # Lepsza stabilność niż LinearRegression dla danych z korelacjami
# current_time_regressor = HuberRegressor() # Odporny na wartości odstające
current_time_regressor = RandomForestRegressor(n_estimators=100, random_state=42) # Dobry dla nieliniowych zależności
# current_time_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # Zazwyczaj bardzo precyzyjny
# current_time_regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1) # Działa dobrze dla nieliniowych danych, ale wolniejszy

# Dla przewidywania oceny jakości:
# current_quality_regressor = LinearRegression()
# current_quality_regressor = Ridge(alpha=1.0)
# current_quality_regressor = HuberRegressor()
current_quality_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# current_quality_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# current_quality_regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

# Lista znanych modeli LLM (do celów predykcji, jeśli nie ma ich w danych treningowych)
# WAŻNE: Ta lista musi zawierać wszystkie modele, których używasz w init_train_data.sh i ask.sh
ALL_KNOWN_MODELS = [
    "gemma3:1b",
    "gemma3:4b",
    "gemma3:12b",
    "aya",
    "aya-expanse",
    "llama2", # Przykład, jeśli dodasz więcej
    "codellama" # Przykład
]

# Kolumny, które będą używane jako cechy (features)
FEATURE_COLUMNS = ['model_name', 'lines', 'words', 'chars']
TARGET_TIME_COLUMN = 'execution_time'
TARGET_QUALITY_COLUMN = 'user_rating'

# Nazwy plików dla zapisanych modeli (czas i jakość)
MODEL_TIME_FILENAME_SUFFIX = "_time_model.pkl"
MODEL_QUALITY_FILENAME_SUFFIX = "_quality_model.pkl"
ENCODER_FILENAME_SUFFIX = "_encoder.pkl"


def train_model(train_filepath, output_model_prefix):
    """
    Trenuje modele regresji dla czasu wykonania i oceny jakości, a następnie zapisuje je do plików.
    """
    try:
        df = pd.read_csv(train_filepath, sep=' ', header=None, comment='#',
                         names=['model_name', 'lines', 'words', 'chars', TARGET_TIME_COLUMN, TARGET_QUALITY_COLUMN])
    except Exception as e:
        print(f"Błąd odczytu pliku treningowego '{train_filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    # Przekonwertuj kolumnę user_rating na typ numeryczny, obsługując '?  jako NaN                                                                 
    df[TARGET_QUALITY_COLUMN] = pd.to_numeric(df[TARGET_QUALITY_COLUMN], errors='coerce')                                                         
                                                                                                  
    # Usuń wiersze, gdzie user_rating jest NaN (czyli było '?')          
    df = df.dropna(subset=[TARGET_QUALITY_COLUMN])                       

    if df.empty:
        print("Brak wystarczających danych do trenowania po odfiltrowaniu niepoprawnych ocen.", file=sys.stderr)
        sys.exit(1)
        
    # Przygotowanie danych
    X = df[FEATURE_COLUMNS]
    y_time = df[TARGET_TIME_COLUMN]
    y_quality = df[TARGET_QUALITY_COLUMN]

    # Preprocessing: One-Hot Encoding dla 'model_name'
    preprocessor = ColumnTransformer(
        transformers=[
            ('model_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['model_name'])
        ],
        remainder='passthrough' # Pozostałe kolumny (lines, words, chars) zostaw bez zmian
    )

    # Pipeline dla czasu wykonania
    time_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', current_time_regressor)
    ])
    
    # Pipeline dla oceny jakości
    quality_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # Używamy tego samego preprocesora
        ('regressor', current_quality_regressor)
    ])

    print(f"Trenowanie modelu czasu wykonania ({type(current_time_regressor).__name__})...")
    time_pipeline.fit(X, y_time)
    print(f"Trenowanie modelu oceny jakości ({type(current_quality_regressor).__name__})...")
    quality_pipeline.fit(X, y_quality)

    # Zapis modeli i preprocesora
    joblib.dump(time_pipeline, output_model_prefix + MODEL_TIME_FILENAME_SUFFIX)
    joblib.dump(quality_pipeline, output_model_prefix + MODEL_QUALITY_FILENAME_SUFFIX)
    
    # Preprocesor jest częścią pipeline'a, więc nie musimy go zapisywać osobno.
    # Ale możemy zapisać sam OneHotEncoder, jeśli chcielibyśmy go użyć w innym miejscu,
    # jednak w tym podejściu cała magia dzieje się w pipeline.
    # Warto jednak zapisać listę znanych modeli, aby upewnić się, że encoder zawsze działa na tym samym zbiorze kategorii.
    
    print(f"Modele regresji (czasu i jakości) oraz preprocesor zapisane do {output_model_prefix}*")
    print(f"Dane użyte do trenowania (po odfiltrowaniu): {len(df)} przypadków.")


def predict_model(model_prefix, lines, words, chars):
    """
    Wczytuje wytrenowane modele i przewiduje czas wykonania oraz ocenę jakości
    dla podanych parametrów i wszystkich znanych modeli.
    """
    time_model_path = model_prefix + MODEL_TIME_FILENAME_SUFFIX
    quality_model_path = model_prefix + MODEL_QUALITY_FILENAME_SUFFIX

    if not os.path.exists(time_model_path) or not os.path.exists(quality_model_path):
        print(f"Błąd: Nie znaleziono plików modeli '{time_model_path}' lub '{quality_model_path}'. "
              "Upewnij się, że model został wytrenowany za pomocą opcji '-t'.", file=sys.stderr)
        sys.exit(1)

    try:
        time_pipeline = joblib.load(time_model_path)
        quality_pipeline = joblib.load(quality_model_path)
    except Exception as e:
        print(f"Błąd wczytywania modeli: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    
    # Przygotuj dane wejściowe dla wszystkich znanych modeli
    for model_name in ALL_KNOWN_MODELS:
        input_data = pd.DataFrame([{
            'model_name': model_name,
            'lines': lines,
            'words': words,
            'chars': chars
        }])
        
        # Przewidywanie czasu wykonania
        predicted_time = time_pipeline.predict(input_data)[0]
        
        # Przewidywanie oceny jakości
        predicted_quality = quality_pipeline.predict(input_data)[0]
        
        # Oceny jakości powinny być w zakresie 0-5. Ograniczamy je.
        predicted_quality = np.clip(predicted_quality, 0, 5)

        results.append({
            'model_name': model_name,
            'predicted_time': predicted_time,
            'predicted_quality': predicted_quality
        })
    
    # Sortowanie wyników (np. od najlepszej jakości do najszybszego, to można dopracować)
    # Na razie po prostu zwracamy je do skryptu bashowego, który zdecyduje
    
    for r in results:
        # Formatowanie wyjścia dla łatwego parsowania w bashu
        print(f"{r['model_name']} {r['predicted_time']:.2f} {r['predicted_quality']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Program do trenowania i przewidywania regresji dla wyboru modelu LLM.")
    parser.add_argument('-t', '--train', nargs=2, metavar=('trainfile', 'modelfile_prefix'),
                        help="Tryb trenowania: <plik_danych_treningowych> <prefiks_pliku_modelu_wyjściowego>")
    parser.add_argument('-a', '--ask', nargs=4, metavar=('modelfile_prefix', 'lines', 'words', 'chars'),
                        help="Tryb zapytania: <prefiks_pliku_modelu> <liczba_linii> <liczba_słów> <liczba_znaków>")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.train:
        train_file = args.train[0]
        model_prefix = args.train[1]
        train_model(train_file, model_prefix)
    elif args.ask:
        model_prefix = args.ask[0]
        try:
            lines = int(args.ask[1])
            words = int(args.ask[2])
            chars = int(args.ask[3])
        except ValueError:
            print("Błąd: Parametry linii, słów i znaków muszą być liczbami całkowitymi.", file=sys.stderr)
            sys.exit(1)
        predict_model(model_prefix, lines, words, chars)

if __name__ == "__main__":
    main()
