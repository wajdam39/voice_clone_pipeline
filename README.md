# Voice Clone Pipeline

Modularny pipeline do klonowania głosu: pobieranie danych audio z YouTube,
preprocessing (resampling, normalizacja, mel-spektrogramy), trenowanie modelu
akustycznego oraz synteza fali audio za pomocą wokodera.

## Struktura projektu

```
voice_clone_pipeline/
├── downloader/         # pobieranie audio z YouTube
│   └── youtube_downloader.py
├── preprocessing/       # przygotowanie danych audio
│   ├── audio_utils.py
│   └── spectrogram.py
├── models/              # definicja modelu, dataset, trening
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── vocoder/             # synteza fali z mel-spektrogramu
│   └── vocoder.py
├── tests/               # testy jednostkowe (pytest), 1:1 z modułami
├── configs/
│   └── config.yaml       # hiperparametry i ścieżki
├── requirements.txt
└── .gitignore
```

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Uruchamianie testów

```bash
pytest
```

## Użycie (szkic)

```python
from downloader.youtube_downloader import download_audio, search_and_download_audio
from preprocessing.audio_utils import load_audio, normalize_audio
from preprocessing.spectrogram import audio_to_melspectrogram
from vocoder.vocoder import melspectrogram_to_audio

# Pobieranie po bezpośrednim linku:
audio_path = download_audio("https://www.youtube.com/watch?v=...", "data/raw")

# Pobieranie na podstawie wyszukanej frazy (np. nazwa utworu/artysty):
audio_path = search_and_download_audio("Artysta - Tytuł utworu", "data/raw")

waveform, sr = load_audio(audio_path)
waveform = normalize_audio(waveform)
mel = audio_to_melspectrogram(waveform, sr)
reconstructed = melspectrogram_to_audio(mel, sr)
```

### `load_audio` i katalog `data/raw`

`preprocessing.audio_utils.load_audio` domyślnie wie o katalogu `data/raw`
(ten sam, do którego domyślnie zapisuje `downloader.youtube_downloader`).
Jeśli podana ścieżka nie istnieje bezpośrednio, funkcja spróbuje znaleźć
plik o tej nazwie w `data/raw`, więc po pobraniu utworu wystarczy podać samą
nazwę pliku:

```python
from downloader.youtube_downloader import search_and_download_audio
from preprocessing.audio_utils import list_audio_files, load_audio

# Pobiera do domyślnego "data/raw"
audio_path = search_and_download_audio("Artysta - Tytuł utworu", "data/raw")

# Można podać pełną ścieżkę zwróconą przez downloader...
waveform, sr = load_audio(audio_path)

# ...albo samą nazwę pliku — zostanie znaleziona w data/raw automatycznie:
waveform, sr = load_audio("Artysta - Tytuł utworu.wav")

# Przetwarzanie wsadowe wszystkich pobranych plików naraz:
for path in list_audio_files():  # skanuje data/raw
    waveform, sr = load_audio(path)
    ...
```

### Uruchamianie preprocessingu z poziomu linii poleceń

Po pobraniu audio do `data/raw` (patrz sekcja niżej) uruchom preprocessing
wszystkich plików naraz za pomocą:

```bash
python -m preprocessing.preprocess
```

Domyślnie skrypt skanuje `data/raw`, dla każdego pliku audio liczy
znormalizowany, log-skalowany mel-spektrogram (parametry wczytywane z
`configs/config.yaml` -> sekcja `preprocessing`) i zapisuje wynik jako
`.npy` w `data/processed`, gotowy do użycia przez `models/train.py`.

Opcjonalne argumenty:

```bash
python -m preprocessing.preprocess --input-dir data/raw --output-dir data/processed --config configs/config.yaml
```

Przykładowy output:
```
Przetworzono 3 plik(ów) audio z 'data/raw':
  -> data/processed/Artysta - Tytuł utworu.npy
  -> ...
```

### Masowe pobieranie wielu utworów na podstawie jednego słowa (`search_and_download_many`)

> ⚠️ **Uwaga prawna:** większość muzyki na YouTube jest chroniona prawem
> autorskim. Pobieranie i wykorzystywanie jej (np. do trenowania modelu
> klonowania głosu) bez zgody może naruszać prawo autorskie oraz Warunki
> Korzystania z YouTube. Używaj tego narzędzia wyłącznie na treściach, do
> których masz prawa, objętych licencją zezwalającą na reuse (np. Creative
> Commons), lub na własnych nagraniach.

Aby na podstawie jednego słowa/frazy pobrać wiele utworów naraz (np. 200),
użyj `search_and_download_many`:

```python
from downloader.youtube_downloader import search_and_download_many

paths = search_and_download_many("fraza wyszukiwania", "data/raw", count=200)
print(f"Pobrano {len(paths)} plików")
```

Albo z CLI:

```bash
python -m downloader.youtube_downloader "fraza wyszukiwania" --count 200
```

Pojedyncze niedostępne/zablokowane filmy są domyślnie pomijane
(`skip_errors=True`), więc jeden błąd nie przerywa całej serii pobierania.

### Statystyki zbioru danych (`compute_npy_statistics` / `generate_dataset_statistics`)

Po przetworzeniu audio na mel-spektrogramy (`data/processed/*.npy`) możesz
wygenerować zbiorcze statystyki opisujące zbiór danych (liczba plików, liczba
klatek czasowych, rozkład wartości mel-spektrogramu itp.) i zapisać je jako
JSON w folderze `data/`.

Najprościej — uruchom sam preprocessing, statystyki generują się automatycznie:

```bash
python -m preprocessing.preprocess
```

Domyślnie zapisze to statystyki do **`data/dataset_stats.json`** zaraz po
przetworzeniu plików z `data/raw` do `data/processed`.

Jeśli chcesz tylko (ponownie) wygenerować statystyki dla już istniejących
plików `.npy` w `data/processed`, bez ponownego przetwarzania audio:

```bash
python -m preprocessing.preprocess --stats-only
```

Możesz też wskazać inne katalogi/ścieżki:

```bash
python -m preprocessing.preprocess --stats-only --output-dir data/processed --stats-output data/dataset_stats.json
```

Aby pominąć generowanie statystyk podczas zwykłego przetwarzania:

```bash
python -m preprocessing.preprocess --no-stats
```

Przykładowa zawartość `data/dataset_stats.json`:

```json
{
  "processed_dir": "data/processed",
  "num_files": 3,
  "n_mels": 80,
  "total_frames": 15234,
  "frames_per_file": {"mean": 5078.0, "min": 4210, "max": 6100, "std": 780.3},
  "value_stats": {"mean": -3.21, "std": 2.05, "min": -11.5, "max": 2.3},
  "files": [
    {"file": "Artysta - Tytuł.npy", "shape": [80, 5210], "mean": -3.1, "std": 2.0, "min": -11.2, "max": 2.1},
    ...
  ]
}
```

Funkcje są też dostępne programistycznie:

```python
from preprocessing.preprocess import generate_dataset_statistics

stats_path = generate_dataset_statistics("data/processed", "data/dataset_stats.json")
print(f"Statystyki zapisane w: {stats_path}")
```

### Pobieranie + preprocessing 200 utworów jedną komendą (`scripts/build_dataset.py`)

Najwygodniejszy sposób na zbudowanie zbioru danych z jednego słowa kluczowego
— jedna komenda pobiera do 200 utworów **i od razu** przetwarza je wszystkie
na mel-spektrogramy `.npy`:

```bash
python -m scripts.build_dataset "fraza wyszukiwania" --count 200
```

Co się dzieje pod spodem:
1. `search_and_download_many("fraza wyszukiwania", "data/raw", count=200)` —
   pobiera do 200 utworów do `data/raw`.
2. `preprocess_directory("data/raw", "data/processed")` — przetwarza
   wszystkie pobrane pliki na log-mel-spektrogramy `.npy` w `data/processed`,
   gotowe do `models/train.py`.

Opcjonalne argumenty:

```bash
python -m scripts.build_dataset "fraza" --count 200 --raw-dir data/raw --processed-dir data/processed --format wav --config configs/config.yaml --fail-fast
```

- `--fail-fast` — przerywa całe pobieranie przy pierwszym błędzie (domyślnie
  błędy pojedynczych filmów są pomijane).

### Pobieranie z poziomu linii poleceń

Moduł `youtube_downloader.py` można też uruchomić bezpośrednio jako skrypt
CLI — wpisujesz frazę, a program wyszukuje najlepsze dopasowanie na YouTube
i pobiera jego audio:

```bash
python -m downloader.youtube_downloader "Artysta - Tytuł utworu"
# lub bez argumentu — zostaniesz zapytany interaktywnie:
python -m downloader.youtube_downloader
```

Opcjonalne argumenty:

```bash
python -m downloader.youtube_downloader "fraza" --output-dir data/raw --format mp3
```

> **Wymaganie:** do konwersji pobranego strumienia na docelowy format audio
> (wav/mp3) wymagany jest zainstalowany i dostępny w PATH systemu **FFmpeg**.

## Roadmapa

- [ ] Implementacja `youtube_downloader.py` (yt-dlp)
- [ ] Implementacja preprocessing audio + mel-spektrogramów
- [ ] Zdefiniowanie architektury modelu akustycznego
- [ ] Pętla treningowa (`train.py`)
- [ ] Integracja wokodera (np. HiFi-GAN)
- [ ] Rozszerzenie testów jednostkowych
