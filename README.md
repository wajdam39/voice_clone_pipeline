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
│   ├── dataset.py        # SegmentedVoiceDataset
│   ├── segmentation.py   # dzielenie spektrogramów na stałe, nakładające się okna
│   ├── unet.py           # architektura 2D U-Net (DDPM epsilon-prediction)
│   ├── diffusion.py      # GaussianDiffusion: noise schedule, q_sample, training_loss, sample
│   ├── model.py           # DiffusionUNet
│   └── train.py
├── vocoder/             # synteza fali z mel-spektrogramu
│   └── hifigan.py        # pretrenowany HiFi-GAN (SpeechBrain / Hugging Face Hub)
├── scripts/             # orkiestracja wielu etapów pipeline'u
│   ├── build_dataset.py  # pobieranie -> preprocessing (2 etapy)
│   ├── generate.py       # wytrenowany model -> generowanie -> synteza (2 etapy)
│   └── run_pipeline.py   # CAŁY pipeline jedną komendą (5 etapów)
├── tests/               # testy jednostkowe (pytest), 1:1 z modułami
├── configs/
│   └── config.yaml       # hiperparametry i ścieżki
├── dvc.yaml              # pipeline DVC (5 etapów, patrz sekcja "Pipeline DVC" niżej)
├── params.yaml           # parametry pipeline'u DVC specyficzne dla uruchomienia
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
from vocoder.hifigan import load_pretrained_vocoder, mel_to_audio

# Pobieranie po bezpośrednim linku:
audio_path = download_audio("https://www.youtube.com/watch?v=...", "data/raw")

# Pobieranie na podstawie wyszukanej frazy (np. nazwa utworu/artysty):
audio_path = search_and_download_audio("Artysta - Tytuł utworu", "data/raw")

waveform, sr = load_audio(audio_path)
waveform = normalize_audio(waveform)
mel = audio_to_melspectrogram(waveform, sr)

vocoder = load_pretrained_vocoder()
reconstructed = mel_to_audio(vocoder, mel)
```

### Pretrenowany wokoder: HiFi-GAN (`vocoder/hifigan.py`)

Projekt integruje **gotowy, wstępnie wytrenowany wokoder HiFi-GAN** — nie
trzeba go trenować od zera.

**Skąd pochodzi model:** `speechbrain/tts-hifigan-ljspeech` z Hugging Face
Hub, wczytywany przez bibliotekę `speechbrain` (`pip install speechbrain`,
już w `requirements.txt`). Przy pierwszym uruchomieniu checkpoint pobiera się
automatycznie i jest cache'owany lokalnie w `pretrained_models/` (katalog
ignorowany przez Git). Ten konkretny checkpoint został wybrany, bo mel-spektrogram,
na którym był trenowany, **dokładnie odpowiada** recepturze używanej w tym
projekcie (`preprocessing.spectrogram.audio_to_melspectrogram`): `sample_rate=22050,
n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0, fmax=8000`,
skala logarytmu naturalnego z podłogą `1e-5`.

**1. Wczytanie modelu:**
```python
from vocoder.hifigan import load_pretrained_vocoder

vocoder = load_pretrained_vocoder()  # pobiera/cache'uje przy pierwszym użyciu
```

**2. Inferencja — konwersja spektrogramu na falę audio:**
```python
from vocoder.hifigan import mel_to_audio
import numpy as np

mel = np.random.randn(80, 512).astype(np.float32)  # [80, 512] albo [B, 80, T] / [B, 1, 80, T]
audio = mel_to_audio(vocoder, mel)  # numpy array: (num_samples,) albo (B, num_samples)
```
`mel_to_audio` automatycznie dopasowuje wymiary wejścia — akceptuje pojedynczy
spektrogram `[80, 512]`, batch `[B, 80, T]`, lub batch z wymiarem kanału
`[B, 1, 80, T]` (dokładnie taki, jaki zwraca `DiffusionUNet`/`GaussianDiffusion.sample`).

**3. Dopasowanie wymiarów i denormalizacja (WAŻNE — inaczej dźwięk będzie zniekształcony):**

Model dyfuzyjny generuje/trenuje się na **znormalizowanych** spektrogramach
(`GaussianDiffusion.sample` zwraca wartości w okolicach zakresu normalizacji,
np. mean≈0/std≈1 dla `zscore`), a wokoder oczekuje **oryginalnej skali
logarytmu naturalnego** mel-energii (tej samej, jaką zwraca
`audio_to_melspectrogram` — **to nie jest skala dB**, tylko `ln(mel_energy)`
z podłogą `1e-5`. Podanie do wokodera spektrogramu wciąż będącego w skali
znormalizowanej (np. wartości bliskie 0) zostanie zinterpretowane jako
`ln(energia) ≈ 0`, czyli energia `e^0 = 1` — realnie bardzo głośny sygnał w
każdym paśmie — co prowadzi do ostrego przesterowania/zniekształcenia dźwięku.

Dlatego przed syntezą trzeba **odwrócić dokładnie tę samą normalizację**,
która została zastosowana przed treningiem, używając **tych samych**
zapisanych statystyk (`data/normalization_stats.json`, patrz sekcja o
normalizacji globalnej wyżej):

```python
from preprocessing.preprocess import load_normalization_stats
from preprocessing.spectrogram import denormalize_melspectrogram_with_stats
from vocoder.hifigan import load_pretrained_vocoder, mel_to_audio

stats = load_normalization_stats("data/normalization_stats.json")
denormalized_mel = denormalize_melspectrogram_with_stats(generated_mel, stats)

vocoder = load_pretrained_vocoder()
audio = mel_to_audio(vocoder, denormalized_mel)
```

Albo jednym wywołaniem (`normalized_mel_to_audio` łączy oba kroki):
```python
from vocoder.hifigan import load_pretrained_vocoder, normalized_mel_to_audio
from preprocessing.preprocess import load_normalization_stats

stats = load_normalization_stats("data/normalization_stats.json")
vocoder = load_pretrained_vocoder()
audio = normalized_mel_to_audio(vocoder, generated_mel, stats)  # denormalizacja + synteza
```

Wzory denormalizacji (dokładna odwrotność `normalize_melspectrogram_with_stats`):
- `zscore`: `mel = normalized * (std + eps) + mean`
- `minmax`: `mel = normalized * (max - min + eps) + min`

**Zapisanie wygenerowanego audio do pliku:**
```python
import soundfile as sf
sf.write("output.wav", audio, samplerate=22050)
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

Filmy dłuższe niż **2 godziny** (np. wielogodzinne sesje/koncerty live) są
domyślnie pomijane (`max_duration_seconds=7200`), żeby przypadkowo nie
ściągnąć bardzo dużych plików, które potem drastycznie wydłużają
preprocessing i trening (jeden godzinny plik audio może wygenerować tysiące
segmentów treningowych). Limit można zmienić lub wyłączyć:

```python
# Zmień limit na 30 minut:
paths = search_and_download_many("fraza", "data/raw", count=200, max_duration_seconds=1800)

# Wyłącz limit całkowicie:
paths = search_and_download_many("fraza", "data/raw", count=200, max_duration_seconds=None)
```

Albo z CLI (`--max-duration` w sekundach; `0` lub wartość ujemna wyłącza limit):

```bash
python -m downloader.youtube_downloader "fraza wyszukiwania" --count 200 --max-duration 1800
```

### Plik cookies i obsługa JS challenge (ważne przy uruchamianiu z EC2/serwera)

YouTube coraz agresywniej blokuje pobieranie z adresów IP centrów danych
(np. AWS EC2) — zarówno przez wymóg zalogowania (cookies), jak i przez
konieczność rozwiązania wyzwania JavaScript (tzw. **EJS**, *External JS
Scripts*) przy niemal każdym żądaniu. `downloader/youtube_downloader.py`
obsługuje oba mechanizmy domyślnie, "out of the box":

**1. Plik cookies** (`youtube.com_cookies.txt`) — uwierzytelnia żądania jako
zalogowana przeglądarka. Wyeksportuj go rozszerzeniem przeglądarki typu
"Get cookies.txt LOCALLY" i umieść w katalogu roboczym projektu (ścieżka
domyślna, konfigurowalna przez `--cookies-file` / `cookies_file=...`).
Plik jest ignorowany przez Git/DVC (zawiera dane uwierzytelniające — patrz
`.gitignore`). Jeśli plik nie istnieje, pobieranie po prostu przebiega
anonimowo bez błędu.

**2. Zewnętrzny runtime JavaScript (EJS)** — yt-dlp potrzebuje uruchomić
skrypty rozwiązujące wyzwanie JS YouTube'a w zewnętrznym silniku JS. Domyślnie
w tym projekcie używany jest **Node.js** (`js_runtime="node"`, zamiast
własnego domyślnego wyboru yt-dlp, czyli `deno` — Node jest zwykle łatwiej
dostępny/instalowalny na typowym obrazie EC2) oraz automatyczne pobieranie
skryptów rozwiązujących wyzwanie z GitHuba (`remote_components=["ejs:github"]`).

Na instancji EC2 (Amazon Linux/Ubuntu) zainstaluj Node.js, jeśli go nie ma:

```bash
# Amazon Linux 2023 / RHEL-based:
sudo dnf install -y nodejs
# Ubuntu/Debian:
sudo apt-get install -y nodejs
```

Bez działającego runtime'u JS pobieranie z adresu IP EC2 może kończyć się
błędem/blokadą nawet przy poprawnym pliku cookies. Oba mechanizmy (cookies +
JS runtime) są niezależne i zalecane razem.

**3. Przerwa między pobraniami (`sleep_interval`/`max_sleep_interval`)** —
przed każdym pobraniem yt-dlp czeka losowy czas (domyślnie od 5 do 15
sekund), żeby ruch nie wyglądał jak zautomatyzowany scraper. Ma to
szczególne znaczenie przy `search_and_download_many` (wiele pobrań z rzędu
z tego samego adresu IP) — bez tej przerwy YouTube znacznie częściej
nakłada rate-limiting/blokadę, zwłaszcza z adresów centrów danych.

**4. Klienci odtwarzacza YouTube (`player_clients`)** — yt-dlp domyślnie
udaje klienta webowego/mobilnej strony (`web`/`mweb`), który jest najściślej
powiązany z wymogiem rozwiązania wyzwania JS i PO Token, i najszybciej trafia
na throttling/blokadę z adresów centrów danych. Domyślnie w tym projekcie
wymuszane są klienci **`ios`** i **`tv_embedded`** (`extractor_args` →
`youtube` → `player_client`), którzy historycznie nie wymagają w ogóle
rozwiązywania wyzwania JS i rzadziej są blokowani — to uzupełnienie (nie
zamiennik) mechanizmów cookies/JS runtime powyżej.

Konfigurowalne z CLI:

```bash
python -m downloader.youtube_downloader "fraza" --count 200 \
    --cookies-file youtube.com_cookies.txt \
    --js-runtime node \
    --remote-components ejs:github \
    --sleep-interval 5 \
    --max-sleep-interval 15 \
    --player-clients ios,tv_embedded
```

Albo z Pythona:

```python
paths = search_and_download_many(
    "fraza", "data/raw", count=200,
    cookies_file="youtube.com_cookies.txt",
    js_runtime="node",            # lub "deno:/path/to/deno", "quickjs", "bun"
    remote_components=["ejs:github"],
    sleep_interval=5,             # minimalna przerwa (sekundy) przed każdym pobraniem
    max_sleep_interval=15,        # maksymalna przerwa (losowa wartość z przedziału)
    player_clients=["ios", "tv_embedded"],
)
```

Żeby wyłączyć któryś z mechanizmów, przekaż pusty string/`None`/wartość
ujemną (`--cookies-file ""`, `--js-runtime ""`, `--remote-components ""`,
`--sleep-interval -1`, `--player-clients ""`). Zob. też
[yt-dlp EJS wiki](https://github.com/yt-dlp/yt-dlp/wiki/ejs) oraz
[yt-dlp YouTube extractor docs](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#youtube).

### Normalizacja spektrogramów: globalna vs lokalna (dlaczego kolejność z segmentacją ma znaczenie)

Pipeline tnie znormalizowane spektrogramy na mniejsze, stałe fragmenty
(`SegmentedVoiceDataset`, 512 ramek) **po** normalizacji. To rodzi pytanie:
jeśli normalizacja liczy `mean`/`std` **per cały plik/utwór**, a do sieci
trafia tylko **wycinek** tego pliku — to wycinek "widzi" statystyki całego
utworu, a nie swoje własne. Fragment z cichej części głośnego utworu
może więc mieć realnie inny rozkład wartości niż fragment z cichego utworu,
mimo że oba formalnie przeszły przez tę samą "normalizację".

**Rozwiązanie: normalizacja globalna (zalecana, domyślna).** Liczymy jedną,
stałą parę `(mean, std)` (lub `(min, max)`) **raz, dla całego zbioru danych**
(`compute_global_mel_stats`), zapisujemy ją (`data/normalization_stats.json`)
i stosujemy **tę samą, stałą transformację** do każdego pliku — niezależnie
od tego, jak głośny jest cały utwór. Dzięki temu normalizacja **komutuje**
z cięciem na fragmenty: *"znormalizuj cały utwór, potem wytnij fragment"*
daje dokładnie ten sam wynik co *"wytnij fragment, potem znormalizuj go tymi
samymi, stałymi liczbami"*. Kolejność przestaje mieć znaczenie.

Normalizacja **lokalna** (per-plik, stare domyślne zachowanie) nie ma tej
własności — każdy plik dostaje własne `mean`/`std`, więc wycięty z niego
fragment jest normalizowany liczbami, które nie opisują jego własnych
statystyk. Zostaje dostępna jako opcja (`scope="local"`), ale **nie jest
zalecana**, gdy dane trafią potem do `SegmentedVoiceDataset`.

**Komenda (zalecany, dwuetapowy workflow):**

```bash
# Krok 1: preprocessing audio -> mel-spektrogramy BEZ normalizacji
python -m preprocessing.preprocess --no-stats

# Krok 2: normalizacja globalna istniejących plików .npy
python -m preprocessing.preprocess --normalize-existing
```

Domyślnie `--normalize-existing` używa `--normalize-scope global` — liczy
`mean`/`std` raz dla wszystkich plików w `data/processed`, zapisuje je do
`data/normalization_stats.json` i stosuje identycznie do każdego pliku,
zapisując wynik w `data/processed_normalized` (oryginały w `data/processed`
pozostają nietknięte).

Opcje:

```bash
# Inna metoda normalizacji:
python -m preprocessing.preprocess --normalize-existing --normalize-method minmax

# Jawne wskazanie zakresu (global jest domyślny i zalecany przed segmentacją):
python -m preprocessing.preprocess --normalize-existing --normalize-scope global

# Stare zachowanie: normalizacja per-plik (NIE zalecane przed segmentacją):
python -m preprocessing.preprocess --normalize-existing --normalize-scope local

# Inna ścieżka na zapisane statystyki globalne:
python -m preprocessing.preprocess --normalize-existing --norm-stats-path data/normalization_stats.json
```

> Flaga `--normalize` (normalizacja "w locie" podczas pierwszego
> przetwarzania audio z `data/raw`) wciąż istnieje, ale zawsze działa w
> trybie lokalnym (per-plik), bo w tym momencie statystyki całego zbioru nie
> są jeszcze znane. Jeśli docelowo dane trafią do `SegmentedVoiceDataset`,
> używaj powyższego dwuetapowego workflow z `--normalize-existing` zamiast
> `--normalize`.

Programistycznie:

```python
from preprocessing.preprocess import (
    compute_global_mel_stats,
    normalize_npy_directory,
    load_normalization_stats,
)
from preprocessing.spectrogram import normalize_melspectrogram_with_stats

# Policz i zastosuj normalizację globalną do całego katalogu naraz:
output_paths = normalize_npy_directory("data/processed", "data/processed_normalized")

# Albo krok po kroku (np. żeby zastosować te same statystyki przy inferencji):
stats = compute_global_mel_stats("data/processed")       # {"method": "zscore", "mean": ..., "std": ...}
normalized_mel = normalize_melspectrogram_with_stats(mel, stats)
```

### Segmentacja spektrogramów na stałe okna (`models.segmentation` / `SegmentedVoiceDataset`)

Po normalizacji spektrogramów warto podzielić je na **stałe, nakładające się
fragmenty czasowe** (np. 512 ramek, krok 256), zamiast podawać do sieci całe
utwory o różnej długości. Dzięki temu każdy przykład treningowy ma identyczny
kształt `[n_mels, segment_length]` (domyślnie `[80, 512]`) i batchowanie w
`DataLoader` działa bez potrzeby pisania własnego `collate_fn`.

**Moduł `models/segmentation.py`** (czysty NumPy, bez zależności od PyTorch):
- `segment_spectrogram(mel, segment_length=512, hop=256)` — dzieli pojedynczy
  spektrogram na listę okien o stałym kształcie; ostatni fragment, krótszy niż
  `segment_length`, jest dopełniany zerami (zero-padding).
- `compute_segment_starts(n_frames, segment_length, hop)` — wylicza same
  indeksy początków okien (z nakładaniem się, gdy `hop < segment_length`).
- `pad_to_length(mel, length)` — samodzielna funkcja do zero-paddingu.
- `build_segment_index(npy_paths, segment_length, hop)` — buduje lekki indeks
  `(ścieżka_pliku, start_ramki)` dla wielu plików naraz, odczytując tylko
  kształty plików (bez wczytywania całych danych do RAM) — wydajne nawet dla
  setek utworów.
- `generate_segments(npy_paths, segment_length, hop)` — leniwy generator
  NumPy jako alternatywa bez PyTorcha.

**Klasa `models.dataset.SegmentedVoiceDataset`** — gotowy `torch.utils.data.Dataset`:

```python
from models.dataset import SegmentedVoiceDataset
from torch.utils.data import DataLoader
from preprocessing.audio_utils import list_audio_files  # lub glob na *.npy

npy_paths = ["data/processed_normalized/utwor1.npy", "data/processed_normalized/utwor2.npy"]

dataset = SegmentedVoiceDataset(npy_paths, segment_length=512, hop=256)
print(len(dataset))        # liczba fragmentów (okien) w całym zbiorze
print(dataset[0].shape)    # torch.Size([80, 512])

loader = DataLoader(dataset, batch_size=16, shuffle=True)
batch = next(iter(loader))
print(batch.shape)         # torch.Size([16, 80, 512])
```

Parametry domyślne (`segment_length=512`, `hop=256`, czyli 50% nakładania się
okien) odpowiadają sekcji `segmentation` w `configs/config.yaml`:

```yaml
segmentation:
  segment_length: 512
  hop: 256
```

`models/train.py` korzysta z `SegmentedVoiceDataset` automatycznie — żadnej
dodatkowej komendy nie trzeba uruchamiać osobno; wystarczy wskazać katalog
przetworzonych (i opcjonalnie znormalizowanych) plików `.npy`:

```bash
python -m models.train --data-dir data/processed_normalized --config configs/config.yaml
```

**Śledzenie eksperymentów (Weights & Biases).** `models/train.py` loguje
metryki treningowe (`train_loss` na epokę, `batch_loss` na batch) do
[Weights & Biases](https://wandb.ai), kontrolowane sekcją `train.wandb`
w `configs/config.yaml`:

```yaml
train:
  wandb:
    enabled: true                     # false wyłącza logowanie (trening działa normalnie)
    project: "voice-clone-diffusion"
    entity: null
    run_name: null
    mode: null                        # "online" / "offline" / "disabled"
```

Jeśli pakiet `wandb` nie jest zainstalowany, logowanie jest automatycznie
pomijane (z ostrzeżeniem w logach), a trening przebiega normalnie bez
dodatkowej konfiguracji. Przed pierwszym użyciem zaloguj się komendą
`wandb login` (lub ustaw `mode: "offline"`, żeby logować lokalnie bez
dostępu do sieci / klucza API).

### Model dyfuzyjny: 2D U-Net (`models/unet.py`, `models/model.py`, `models/diffusion.py`)

Głównym modelem akustycznym projektu jest **model dyfuzyjny (DDPM)** generujący
mel-spektrogramy o stałym kształcie `[80, 512]` (dokładnie kształt zwracany
przez `SegmentedVoiceDataset`).

**1. Architektura `UNet2D` (`models/unet.py`)**
- Wejście/wyjście: `[batch, 1, n_mels, segment_length]` (domyślnie `[B, 1, 80, 512]`).
- `SinusoidalPositionEmbeddings` koduje krok dyfuzji `t`, który następnie
  przechodzi przez 2-warstwowy MLP i jest wstrzykiwany do **każdego** bloku
  rezydualnego (dodawany do map cech po pierwszej konwolucji).
- `ResidualBlock`: GroupNorm → SiLU → Conv2d, z projekcją time-embeddingu i
  połączeniem rezydualnym.
- Encoder (3 etapy, kanały `[32, 64, 128]` domyślnie) ze stride-2 konwolucjami
  (`Downsample`) między etapami, bottleneck (2 bloki rezydualne), decoder
  (`Upsample` + konkatenacja ze skip connection z odpowiadającego etapu encodera).
- Asymetria wymiarów (80 vs 512) jest obsłużona przez downsamplowanie **obu**
  wymiarów równocześnie — przy domyślnych 3 etapach kanałów oba wymiary muszą
  być podzielne przez 4 (80→40→20, 512→256→128), co jest walidowane w konstruktorze.

**2. Harmonogram szumu i proces forward (`models/diffusion.py` — `GaussianDiffusion`)**
- `linear_beta_schedule` — liniowy harmonogram wariancji szumu `beta_t` od
  `beta_start=1e-4` do `beta_end=0.02` na przestrzeni `num_timesteps` (domyślnie 1000) kroków.
- `q_sample(x_start, t, noise=None)` — dodaje szum gaussowski do czystego
  spektrogramu w zamkniętej formie: `x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise`.

**3. Strata treningowa (`GaussianDiffusion.training_loss`)**
```python
def training_loss(self, x_start, t=None):
    if t is None:
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
    x_t, noise = self.q_sample(x_start, t)          # dodaj szum w losowym kroku t
    predicted_noise = self.model(x_t, t)            # U-Net przewiduje szum
    return F.mse_loss(predicted_noise, noise)        # MSE(szum_prawdziwy, szum_przewidziany)
```
Dokładnie to wywołuje `models/train.py` w każdej iteracji pętli treningowej.

**4. Procedura inferencji / generowania (`GaussianDiffusion.sample`)**
```python
@torch.no_grad()
def sample(self, shape=(1, 1, 80, 512), device=None):
    x = torch.randn(shape, device=device)            # start: czysty szum gaussowski
    for t in reversed(range(self.num_timesteps)):     # T-1, T-2, ..., 0
        x = self.p_sample(x, t)                       # jeden krok odszumiania
    return x                                           # wygenerowany spektrogram
```

**Przykład użycia (trening i generowanie):**
```python
from models.model import DiffusionUNet
from models.diffusion import GaussianDiffusion
import torch

model = DiffusionUNet(n_mels=80, segment_length=512)
diffusion = GaussianDiffusion(model, num_timesteps=1000)

# --- Trening (jeden krok) ---
x_start = torch.randn(16, 1, 80, 512)   # batch znormalizowanych spektrogramów
loss = diffusion.training_loss(x_start)  # MSE(szum, szum_przewidziany)
loss.backward()

# --- Generowanie nowego spektrogramu ---
generated = diffusion.sample(shape=(1, 1, 80, 512))  # [1, 1, 80, 512]
```

**Uruchomienie treningu** (automatycznie korzysta z `DiffusionUNet` + `GaussianDiffusion`,
parametry z sekcji `model`/`diffusion` w `configs/config.yaml`):
```bash
python -m models.train --data-dir data/processed_normalized --config configs/config.yaml
```

### Generowanie audio z wytrenowanego modelu (`scripts/generate.py`)

Po zakończonym treningu (checkpoint `.pt` + zapisane statystyki normalizacji)
jedna komenda generuje nowy spektrogram i syntetyzuje z niego plik `.wav`:

```bash
python -m scripts.generate --checkpoint checkpoints/model_final.pt --output output/generated.wav
```

Pod spodem: `models.train.load_trained_model` (odtwarza architekturę z configu
i wczytuje wagi) → `GaussianDiffusion.sample()` (odszumianie od czystego
szumu) → `denormalize_melspectrogram_with_stats` (odwrócenie normalizacji
zapisanymi statystykami) → pretrenowany HiFi-GAN (`mel_to_audio`) → zapis `.wav`.

Opcje: `--num-samples N` (wygeneruj N niezależnych klipów), `--config`,
`--norm-stats-path`, `--sample-rate`.

### Cały pipeline jedną komendą (`scripts/run_pipeline.py`)

Najważniejsza komenda w projekcie — łączy **wszystkie 5 etapów** w jedno
wywołanie: **pobieranie → przygotowanie → trenowanie → generowanie → przywracanie**:

```bash
python -m scripts.run_pipeline "słowo kluczowe" --count 50 --epochs 20
```

Co robi, krok po kroku:
1. **Pobieranie** — `search_and_download_many("słowo kluczowe", ..., count=50)` → `data/raw`.
2. **Przygotowanie** — `preprocess_directory` (audio → mel-spektrogramy) →
   `normalize_npy_directory(scope="global")` (normalizacja całego zbioru,
   zgodna z późniejszą segmentacją) → `data/processed_normalized` +
   `data/normalization_stats.json`.
3. **Trenowanie** — `models.train.train()` na `SegmentedVoiceDataset` →
   `checkpoints/model_final.pt`.
4. **Generowanie** — `GaussianDiffusion.sample()`: nowy spektrogram od czystego szumu.
5. **Przywracanie** — denormalizacja + synteza pretrenowanym HiFi-GAN → `output/generated.wav`.

> ⚠️ **Uwaga prawna:** etap pobierania może naruszać prawa autorskie, jeśli
> użyty na chronionej muzyce bez zgody — patrz ostrzeżenie w sekcji o
> `search_and_download_many` powyżej. Używaj tylko na treściach, do których
> masz prawa.

**Pomijanie etapów** — przydatne przy iteracyjnej pracy (np. trening bez
ponownego pobierania, albo samo wygenerowanie nowego audio z już wytrenowanego
modelu):

```bash
# Dane już pobrane i przygotowane — od razu trenuj i generuj:
python -m scripts.run_pipeline "fraza" --skip-download --skip-preprocess --epochs 50

# Model już wytrenowany — wygeneruj tylko nowe audio:
python -m scripts.run_pipeline "fraza" --skip-download --skip-preprocess --skip-train

# Tylko pobierz i przygotuj dane, bez trenowania/generowania:
python -m scripts.run_pipeline "fraza" --count 100 --skip-train --skip-generate
```

Pełna lista opcji: `--raw-dir`, `--processed-dir`, `--normalized-dir`,
`--norm-stats-path`, `--config`, `--output`, `--format`, `--fail-fast`,
`--max-duration` (limit długości pojedynczego filmu w sekundach, domyślnie
7200 = 2 godziny; `0` lub ujemna wartość wyłącza limit — patrz sekcja o
`search_and_download_many` powyżej).

### Pipeline DVC (`dvc.yaml`, `params.yaml`)

Te same 5 etapów co w `scripts/run_pipeline.py` są też zdefiniowane jako
**pipeline [DVC](https://dvc.org)** (`dvc.yaml`), podzielony na 5 niezależnie
cache'owalnych stage'ów: `download → preprocess → normalize → train → generate`.
Dzięki temu `dvc repro` uruchamia **tylko** te etapy, których zależności
(kod, dane wejściowe lub parametry z `configs/config.yaml`/`params.yaml`)
faktycznie się zmieniły — reszta jest odtwarzana z cache'u DVC.

**Instalacja i inicjalizacja** (już wykonane w tym repo — `dvc init` tworzy
katalog `.dvc/`):
```bash
pip install dvc
dvc init
```

**Uruchomienie całego pipeline'u:**
```bash
dvc repro
```

**Typowe komendy:**
```bash
dvc repro train              # uruchom tylko do (włącznie z) etapu train
dvc repro -f download        # wymuś ponowne uruchomienie etapu download
dvc dag                      # wizualizacja grafu etapów (ASCII)
dvc status                   # które etapy są "nieaktualne" i dlaczego
dvc metrics show             # pokaż metrics/train_metrics.json, data/dataset_stats.json
dvc metrics diff             # porównaj metryki między commitami/branchami
dvc plots show               # wykres metrics/train_loss.csv (train_loss vs epoch)
dvc params diff               # porównaj parametry (configs/config.yaml, params.yaml)
```

**Eksperymentowanie z nadpisywaniem parametrów** (bez edycji plików):
```bash
dvc exp run -S train.num_epochs=5 -S download.query="inna fraza"
dvc exp show
```

**Struktura:**
- `dvc.yaml` — definicja 5 etapów (deps/outs/params/metrics/plots).
- `params.yaml` — parametry specyficzne dla *uruchomienia* pipeline'u
  (fraza wyszukiwania YouTube, liczba próbek do wygenerowania), których nie
  ma w `configs/config.yaml`. Hiperparametry modelu/treningu/preprocessingu
  są śledzone bezpośrednio z `configs/config.yaml` (sekcja `params:` każdego
  stage'a wskazuje konkretne klucze, np. `train.num_epochs`).
- `dvc.lock` — generowany automatycznie przez `dvc repro`/`dvc exp run`;
  zawiera hashe zależności/wyjść każdego etapu — commituj go do Git.
- `metrics/train_metrics.json`, `metrics/train_loss.csv`,
  `data/dataset_stats.json` — pliki metryk/wykresów (`cache: false` w
  `dvc.yaml`, więc trafiają bezpośrednio do Git zamiast do cache'u DVC —
  są małe i warto je móc zdiffować commit-do-commitu).
- Duże/binarne wyjścia (`data/raw`, `data/processed`,
  `data/processed_normalized`, `checkpoints/model_final.pt`, `output/`)
  są cache'owane przez DVC i wykluczone z Git (patrz `.gitignore`).

> ⚠️ Ta sama uwaga prawna co dla `scripts/run_pipeline.py` dotyczy etapu
> `download` w `dvc.yaml` — patrz wyżej.

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
- `--max-duration` — limit długości pojedynczego filmu w sekundach (domyślnie
  7200 = 2 godziny, patrz sekcja o `search_and_download_many` powyżej).

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

- [x] Implementacja `youtube_downloader.py` (yt-dlp)
- [x] Implementacja preprocessing audio + mel-spektrogramów
- [x] Zdefiniowanie architektury modelu akustycznego (2D U-Net + dyfuzja DDPM)
- [x] Pętla treningowa (`train.py`)
- [x] Integracja wokodera (pretrenowany HiFi-GAN, `vocoder/hifigan.py`)
- [x] Rozszerzenie testów jednostkowych
- [x] Orkiestracja całego pipeline'u jedną komendą (`scripts/run_pipeline.py`)
