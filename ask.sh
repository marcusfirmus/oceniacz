#!/bin/bash
export LC_NUMERIC=C # Wymuś użycie kropki jako separatora dziesiętnego dla całego skryptu

# --- Konfiguracja ścieżek i parametrów ---
PROJ_DIR="/home/marek/work/ask"
DATA_DIR="$PROJ_DIR/data"
MODEL_PREFIX="$PROJ_DIR/model/current_model"
TRAIN_DATA_FILE="$DATA_DIR/train_data.txt"
REGRESSION_SCRIPT="$PROJ_DIR/regression.py"

# --- Kryteria wyboru modelu ---
MIN_ACCEPTABLE_QUALITY=2.5
SPEED_PENALTY_FACTOR=0.05

# --- Lista znanych modeli LLM (musi być zgodna z ALL_KNOWN_MODELS w regression.py) ---
KNOWN_MODELS=(
    "gemma3:1b"
    "gemma3:4b"
    "gemma3:12b"
    "aya"
    "aya-expanse"
    "llama2"
    "codellama"
)

# --- Upewnij się, że katalogi istnieją ---
mkdir -p "$DATA_DIR"
mkdir -p "$PROJ_DIR/model"

# --- Inicjalizacja pliku danych treningowych, jeśli nie istnieje ---
if [ ! -f "$TRAIN_DATA_FILE" ]; then
    echo "# model lines words chars execution_time user_rating" > "$TRAIN_DATA_FILE"
    if [ -f "$DATA_DIR/initial_train_data.txt" ]; then
        cat "$DATA_DIR/initial_train_data.txt" >> "$TRAIN_DATA_FILE"
        echo "Skopiowano dane z initial_train_data.txt do $TRAIN_DATA_FILE"
    fi
fi

# --- Funkcja pomocnicza do mierzenia wc ---
get_wc_stats() {
    local text="$1"
    echo "$text" | wc -l -w -m | awk '{print $1, $2, $3}'
}

# --- Główna logika skryptu ---

echo "Cześć! Jestem Twoim lokalnym asystentem AI."

echo -n "Jak mogę Ci pomóc? (Wpisz zadanie i naciśnij Enter): "
read -r USER_PROMPT

if [ -z "$USER_PROMPT" ]; then
    echo "Nie podałeś żadnego zadania. Do zobaczenia!"
    exit 0
fi

read -r LINES WORDS CHARS <<< $(get_wc_stats "$USER_PROMPT")
echo "Twoje zapytanie: Linie=$LINES, Słowa=$WORDS, Znaki=$CHARS"

echo "Analizuję zapytanie i wybieram najlepszy model..."

if [ ! -f "${MODEL_PREFIX}_time_model.pkl" ] || [ ! -f "${MODEL_PREFIX}_quality_model.pkl" ]; then
    echo "Modele regresji nie istnieją. Próbuję je wytrenować po raz pierwszy..."
    python3 "$REGRESSION_SCRIPT" -t "$TRAIN_DATA_FILE" "$MODEL_PREFIX" || { echo "Błąd: Nie udało się wytrenować modeli regresji. Upewnij się, że '$TRAIN_DATA_FILE' zawiera dane i 'regression.py' działa poprawnie."; exit 1; }
fi

PREDICTIONS_RAW=$(python3 "$REGRESSION_SCRIPT" -a "$MODEL_PREFIX" "$LINES" "$WORDS" "$CHARS")
if [ $? -ne 0 ]; then
    echo "Błąd podczas uzyskiwania predykcji. Sprawdź plik '$REGRESSION_SCRIPT' i ścieżkę modelu '$MODEL_PREFIX'." >&2
    echo "Wytrenuj model najpierw: python3 $REGRESSION_SCRIPT -t $TRAIN_DATA_FILE $MODEL_PREFIX" >&2
    exit 1
fi

BEST_MODEL=""
BEST_EFFECTIVE_SCORE=-1000000.0

echo "Przewidywane wyniki dla modeli:"
echo "Model         Czas [s]    Jakość     Wynik decyzyjny"
echo "----------------------------------------------------"

while IFS= read -r line; do
    MODEL_NAME=$(echo "$line" | awk '{print $1}')
    # Użyj 'tr' do zamiany przecinków na kropki na wszelki wypadek, choć python3 już je wypisuje
    PREDICTED_TIME=$(echo "$line" | awk '{print $2}' | tr ',' '.')
    PREDICTED_QUALITY=$(echo "$line" | awk '{print $3}' | tr ',' '.')

    if (( $(echo "$PREDICTED_QUALITY < $MIN_ACCEPTABLE_QUALITY" | bc -l) )); then
        printf "%-13s %-10.2f %-10.2f Odrzucony (niska jakość)\n" "$MODEL_NAME" "$PREDICTED_TIME" "$PREDICTED_QUALITY"
        continue
    fi
    
    EFFECTIVE_SCORE=$(echo "$PREDICTED_QUALITY - ($PREDICTED_TIME * $SPEED_PENALTY_FACTOR)" | bc -l)
    
    # Wyświetl wynik decyzyjny, używając printf jako osobnej komendy
    # Prawdopodobnie EFFECTIVE_SCORE z bc -l też może mieć przecinek, jeśli locale nie zadziałało, więc tr
    FORMATTED_EFFECTIVE_SCORE=$(echo "$EFFECTIVE_SCORE" | tr ',' '.')
    printf "%-13s %-10.2f %-10.2f %.2f\n" "$MODEL_NAME" "$PREDICTED_TIME" "$PREDICTED_QUALITY" "$FORMATTED_EFFECTIVE_SCORE"

    if (( $(echo "$EFFECTIVE_SCORE > $BEST_EFFECTIVE_SCORE" | bc -l) )); then
        BEST_EFFECTIVE_SCORE="$EFFECTIVE_SCORE"
        BEST_MODEL="$MODEL_NAME"
    fi
done <<< "$PREDICTIONS_RAW"

if [ -z "$BEST_MODEL" ]; then
    echo "Nie udało się znaleźć odpowiedniego modelu spełniającego kryteria ($MIN_ACCEPTABLE_QUALITY jakości)." >&2
    echo "Rozważ złagodzenie kryteriów w skrypcie $0 lub dodanie więcej danych treningowych." >&2
    exit 1
fi

echo "----------------------------------------------------"
# Tutaj zmieniamy sposób wyświetlania, żeby uniknąć formatowania w parametrach Basha
FORMATTED_BEST_EFFECTIVE_SCORE=$(echo "$BEST_EFFECTIVE_SCORE" | tr ',' '.' | cut -d'.' -f1,2 | cut -c -5) # Przycinanie do 5 znaków, żeby pasowało do .2f
printf "Wybrano model: %s (wynik decyzyjny: %.2f)\n" "$BEST_MODEL" "$FORMATTED_BEST_EFFECTIVE_SCORE"


echo "Generowanie odpowiedzi przez $BEST_MODEL..."
start_time=$(date +%s.%N)
ollama stop "$BEST_MODEL" # Dodaj unload, jeśli nie jest już tam
MODEL_RESPONSE=$(ollama run "$BEST_MODEL" "$USER_PROMPT")
end_time=$(date +%s.%N)

ACTUAL_EXECUTION_TIME=$(echo "$end_time - $start_time" | bc -l | tr ',' '.')

printf "%s--- ODPOWIEDŹ %s (czas: %.2fś) ---\n" '' "$BEST_MODEL" "$ACTUAL_EXECUTION_TIME"
echo "$MODEL_RESPONSE"
echo "----------------------------------------------------"

USER_RATING=""
while [[ ! "$USER_RATING" =~ ^[0-5]$ && "$USER_RATING" != "?" ]]; do
    echo -n "Oceń jakość odpowiedzi (0-5, '?' aby pominąć w statystykach): "
    read -r USER_RATING
done

if [ "$USER_RATING" != "?" ]; then
    echo "$BEST_MODEL $LINES $WORDS $CHARS $ACTUAL_EXECUTION_TIME $USER_RATING" >> "$TRAIN_DATA_FILE"
    echo "Dane zapisane do '$TRAIN_DATA_FILE'."
else
    echo "Ocena pominięta. Dane nie zostały zapisane do treningu."
fi

echo "Przetrenowywuję modele regresji z aktualnymi danymi..."
python3 "$REGRESSION_SCRIPT" -t "$TRAIN_DATA_FILE" "$MODEL_PREFIX" || { echo "Ostrzeżenie: Nie udało się ponownie wytrenować modeli regresji."; }
echo "Gotowe!"

exit 0
