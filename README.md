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
