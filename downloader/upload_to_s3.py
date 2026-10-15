import os
import boto3
from botocore.exceptions import NoCredentialsError


def upload_folder_to_s3(
    local_dir="data/raw", bucket_name="TWOJA_NAZWA_BUCKETU", s3_prefix="audio/raw"
):
  # Inicjalizacja klienta S3
  s3_client = boto3.client("s3")

  if not os.path.exists(local_dir):
    print(f"Błąd: Folder lokalny '{local_dir}' nie istnieje!")
    return

  wav_files = [f for f in os.listdir(local_dir) if f.lower().endswith(".wav")]

  if not wav_files:
    print(f"Brak plików WAV w folderze '{local_dir}' do wysłania.")
    return

  print(
      f"Rozpoczynam wysyłanie {len(wav_files)} plików WAV do S3"
      f" (bucket: {bucket_name}, ścieżka: {s3_prefix}/)..."
  )

  success_count = 0

  for i, filename in enumerate(wav_files, 1):
    local_path = os.path.join(local_dir, filename)
    # Tworzymy ścieżkę docelową w S3 (klucz)
    s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename
    # Zamiana backslashów na slash dla poprawności ścieżek S3 na Windowsie
    s3_key = s3_key.replace("\\", "/")

    try:
      print(f"[{i}/{len(wav_files)}] Wysyłanie: {filename} -> s3://{bucket_name}/{s3_key}")
      
      # Przesyłanie pliku
      s3_client.upload_file(local_path, bucket_name, s3_key)
      success_count += 1
      
    except NoCredentialsError:
      print("Błąd: Nie znaleziono poświadczeń AWS. Skonfiguruj je poleceniem 'aws configure'.")
      break
    except Exception as e:
      print(f"-> Błąd podczas wysyłania {filename}: {e}")

  print(
      "\nPrzesyłanie zakończone!"
      f" Pomyślnie wysłano {success_count}/{len(wav_files)} plików do S3."
  )


if __name__ == "__main__":
  # --- SKONFIGURUJ SWÓJ BUCKET TUTAJ ---
  BUCKET_NAME = "voice-clone-dvc-storage-wajda"  # Wpisz dokładną nazwę swojego bucketu
  S3_FOLDER_PREFIX = "data/raw"    # Folder wewnątrz bucketu (opcjonalnie)

  upload_folder_to_s3(
      local_dir="data/raw", 
      bucket_name=BUCKET_NAME, 
      s3_prefix=S3_FOLDER_PREFIX
  )