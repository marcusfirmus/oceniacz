# Oceniacz – eksperyment z AI i `ollama`

To repozytorium zawiera pliki związane z artykułem na [moim blogu](https://marcusfirmus.github.io/)  
Opisuję w nim eksperyment, w którym przy pomocy modelu językowego stworzyłem narzędzie oceniające trudność zapytań i wybierające odpowiedni model LLM w `ollama`.

## Zawartość

- **[`FirstPrompt.txt`](FirstPrompt.txt)** – pierwsze zapytanie/specyfikacja w języku naturalnym, która zapoczątkowała cały projekt.  
- **[`regression.py`](regression.py)** – program w Pythonie, trenujący regresory (`scikit-learn`) i dokonujący predykcji.  
- **[`init_train_data.sh`](init_train_sata.sh)** – skrypt do przygotowania początkowych danych treningowych (egzaminowanie modeli i ręczna ocena jakości odpowiedzi).  
- **[`ask.sh`](ask.sh)** – główny skrypt użytkownika: mierzy zapytanie, przewiduje trudność, wybiera model i zbiera oceny.  

## Jak zacząć

1. **Przygotuj dane treningowe**  
   Uruchom:
   ```bash
   ./init_train_data.sh
````

Skrypt wygeneruje przykładowe zapytania i odpowiedzi modeli. Następnie oceń odpowiedzi w pliku wynikowym zgodnie z instrukcją w komentarzu w skrypcie.

2. **Wytrenuj modele regresji**

   ```bash
   python3 regression.py -t data/initial_train_data.txt model/initial_model
   ```

3. **Zadaj własne pytanie**

   ```bash
   ./ask.sh
   ```

   Skrypt wybierze odpowiedni model, uruchomi go przez `ollama`, wyświetli wynik i poprosi Cię o ocenę jakości. Dane zostaną zapisane i użyte do ponownego treningu regresora.

## Więcej informacji

Szczegółowy opis eksperymentu znajdziesz w artykule:
👉 [Oceniacz modeli OLLAMA](https://marcusfirmus.github.io/Oceniacz.html)

---

> **Uwaga:** Projekt jest eksperymentalny – to raczej „zabawka” niż gotowe narzędzie produkcyjne.
> Powstał, by zademonstrować dialogowe, iteracyjne tworzenie kodu przy pomocy modeli językowych.
