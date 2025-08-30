#!/bin/bash

# --- Konfiguracja ścieżek ---
# WAŻNE: Upewnij się, że ta ścieżka jest zgodna z planowaną ścieżką projektu!
PROJ_DIR="/home/marek/work/ask"
DATA_DIR="$PROJ_DIR/data" # Utworzymy katalog na dane
TRAIN_DATA_FILE="$DATA_DIR/initial_train_data.txt"
MODEL_PREFIX="$PROJ_DIR/model/initial_model" # Prefiks dla zapisywanych modeli

# --- Lista modeli do wstępnego egzaminowania ---
MODELS=(
    "gemma3:1b"
    "gemma3:4b"
    "gemma3:12b"
    "aya"
    "aya-expanse"
)

# --- Przykładowe zadania (minimum 3, nie więcej niż 8 na model, różnorodność!) ---
# Pamiętaj o używaniu pojedynczych cudzysłowów dla zapytań, aby uniknąć interpretacji przez bash
TASKS=(
"Ile wynosi 100 razy 919.2?  Podaj tylko wynik."
"Mój linux nie chce się zupgradować, dostaję 
 dziwne niezrozumiałe komunikaty 
 - chyba po angielsku. Co mam robić? Poradź mi!"
"Napisz krótką historię o odważnym królu i smoku, który lubił czytać książki."
"Podaj listę 5 najlepszych książek science-fiction, 
 które każdy powinien przeczytać."
"Wyjaśnij pojęcie 'machine learning' 
 w prostych słowach, tak 
 aby zrozumiało je dziecko w wieku 10 lat."
"Napisz krótki wiersz o jesieni w trzech zwrotkach."
"Jakie są główne różnice między programowaniem obiektowym a funkcyjnym? 
 Użyj przykładów w Pythonie."
"Podaj przepis na prostą pizzę domową."
"Wytłumacz, dlaczego koty mruczą."
"Stwórz plan nauki Pythona od podstaw 
 do poziomu zaawansowanego dla osoby, 
 która ma 2 godziny dziennie."
)

# --- Funkcja pomocnicza do mierzenia wc ---
get_wc_stats() {
    local text="$1"
    echo "$text" | wc -l -w -m | awk '{print $1, $2, $3}' # lines, words, chars
}

# --- Instrukcje dla człowieka (czyli Ciebie) ---
read -r -d '' INSTRUCTIONS << EOM
#####################################################################################
#                                INSTRUKCJE DLA UŻYTKOWNIKA                        #
#####################################################################################
#
# Ten skrypt wygeneruje wstępne dane treningowe dla modeli regresji.
# Dla każdego zapytania i modelu LLM zostanie wykonane zapytanie.
# Wyniki (model, statystyki 'wc', czas wykonania) zostaną zapisane w pliku:
#   $TRAIN_DATA_FILE
#
# Oprócz tych danych, każdy wiersz w pliku będzie miał puste miejsce na końcu
# na Twoją OCENĘ jakości odpowiedzi.
#
# JAK OCENIĆ ODPOWIEDZI:
# 1. Po zakończeniu działania tego skryptu, OTWÓRZ plik '$TRAIN_DATA_FILE' w edytorze tekstowym.
# 2. Dla KAŻDEGO wiersza w pliku zobaczysz:
#    "nazwa_modelu liczba_linii liczba_słów liczba_znaków czas_wykonania OCENA_TUTAJ"
# 3. Przejrzyj zapisane odpowiedzi w katalogu '$DATA_DIR/responses/'.
#    Dla każdego pliku odpowiedzi (np. gemma3:1b_task_1.txt) znajdziesz odpowiedź na konkretne pytanie.
# 4. Oceń jakość odpowiedzi, przypisując jej jedną z wartości:
#    - 0: Bardzo słaba, bezużyteczna odpowiedź.
#    - 1: Słaba, zawiera błędy lub jest daleka od oczekiwań.
#    - 2: Średnia, ale wymaga poprawek lub jest niekompletna.
#    - 3: Dobra, zadowalająca odpowiedź.
#    - 4: Bardzo dobra, spełnia oczekiwania.
#    - 5: Doskonała, precyzyjna, kompletna, kreatywna.
#    - ?: Jeśli z jakiegoś powodu nie chcesz uwzględniać tej odpowiedzi w treningu (np. błąd modelu, niezrozumiałe pytanie).
#      Wiersze z '?' zostaną pominięte podczas trenowania.
# 5. Wpisz swoją ocenę w miejsce "OCENA_TUTAJ" w pliku '$TRAIN_DATA_FILE'.
#    Pamiętaj, aby oddzielić ją spacją od 'czas_wykonania'.
#
# Przykład po edycji:
# gemma3:1b 12 83 182 13.4 3
# mistral 12 83 182 103.21 2
#
# PO ZAKOŃCZENIU OCENIANIA:
# Uruchom wstępne trenowanie modelu regresji. Przejdź do katalogu projektu:
#   cd $PROJ_DIR
# Następnie uruchom:
#   python3 regression.py -t $TRAIN_DATA_FILE $MODEL_PREFIX
#
# To utworzy pliki modeli w katalogu '$PROJ_DIR/model/'.
# Po tym kroku system będzie gotowy do użycia z 'ask.sh'.
#####################################################################################
EOM

echo "$INSTRUCTIONS"

# --- Przygotowanie katalogów ---
mkdir -p "$DATA_DIR/responses"
mkdir -p "$PROJ_DIR/model"
rm -f "$TRAIN_DATA_FILE" # Usuwamy stary plik, jeśli istnieje

# --- Generowanie danych ---
echo "Generowanie wstępnych danych treningowych..."

task_counter=0
for task_prompt in "${TASKS[@]}"; do
    task_counter=$((task_counter + 1))
    echo "Task $task_counter: '$task_prompt'"

    # Oblicz statystyki wc dla promptu
    wc_stats=$(get_wc_stats "$task_prompt") # lines words chars

    for model_name in "${MODELS[@]}"; do
        echo "  Egzaminowanie modelu: $model_name..."

        response_file="$DATA_DIR/responses/${model_name//[:\/]/_}_task_${task_counter}.txt"
        
        # Używamy mktemp do bezpiecznego tworzenia tymczasowego pliku na stdout/stderr
        TEMP_OUTPUT=$(mktemp)

        # Uruchom ollama i zmierz czas
        # Przekierowujemy stderr do TEMP_OUTPUT, aby złapać 'real' czas
        # stdout (odpowiedź modelu) również do TEMP_OUTPUT
        # Następnie parsujemy czas z TEMP_OUTPUT
        start_time=$(date +%s.%N)
        ollama run "$model_name" "$task_prompt" > "$response_file" 2> "$TEMP_OUTPUT"
        end_time=$(date +%s.%N)

        # Czas rzeczywisty
        execution_time=$(echo "$end_time - $start_time" | bc -l)
        
        # Dodajemy do pliku treningowego (z placeholderem na ocenę)
        echo "$model_name $wc_stats $execution_time ?" >> "$TRAIN_DATA_FILE"
        echo "    -> Czas: ${execution_time}s. Odpowiedź zapisana do: $response_file"
        
        rm "$TEMP_OUTPUT" # Usuwamy tymczasowy plik

        ollama stop "$model_name"
        sleep 5
        echo "$model_name" usunięty z pamięci.
        sleep 1
    done
done

echo ""
echo "--------------------------------------------------------"
echo "Proces generowania wstępnych danych zakończony."
echo "Teraz postępuj zgodnie z INSTRUKCJAMI POWYŻEJ, aby ocenić odpowiedzi i wytrenować modele."
echo "Plik do edycji: $TRAIN_DATA_FILE"
echo "Katalog z odpowiedziami: $DATA_DIR/responses/"
echo "--------------------------------------------------------"
