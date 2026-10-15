import os
from pydub import AudioSegment


def convert_mp3_to_wav(
    input_dir="cc_audio_downloads", output_dir="data/raw"
):
  # Tworzenie folderu docelowego, jeśli nie istnieje
  os.makedirs(output_dir, exist_ok=True)

  if not os.path.exists(input_dir):
    print(f"Błąd: Folder źródłowy '{input_dir}' nie istnieje!")
    return

  mp3_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp3")]

  if not mp3_files:
    print(f"Brak plików MP3 w folderze '{input_dir}'.")
    return

  print(
      f"Znaleziono {len(mp3_files)} plików MP3. Rozpoczynam konwersję do"
      f" '{output_dir}'..."
  )

  success_count = 0

  for i, filename in enumerate(mp3_files, 1):
    input_path = os.path.join(input_dir, filename)
    # Zmiana rozszerzenia z .mp3 na .wav
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}.wav"
    output_path = os.path.join(output_dir, output_filename)

    try:
      print(
          f"[{i}/{len(mp3_files)}] Konwertowanie: {filename} ->"
          f" {output_filename}"
      )

      # Wczytaj MP3 i wyeksportuj do WAV
      audio = AudioSegment.from_mp3(input_path)
      audio.export(output_path, format="wav")

      success_count += 1
    except Exception as e:
      print(f"-> Błąd podczas konwersji pliku {filename}: {e}")

  print(
      "\nKonwersja zakończona!"
      f" Pomyślnie przekonwertowano {success_count}/{len(mp3_files)} plików do"
      f" '{output_dir}'."
  )


if __name__ == "__main__":
  convert_mp3_to_wav()