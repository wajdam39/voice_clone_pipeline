import os
import time
import requests


def download_from_archive_api(music_type="ambient", target_count=200, output_dir="cc_audio_downloads"):
  os.makedirs(output_dir, exist_ok=True)
  
  collection_identifier = "netlabels"
  headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
  }
  
  print(f"Pobieranie listy utworów dla typu '{music_type}' z kolekcji '{collection_identifier}'...")
  
  # Zapytanie do API Internet Archive z filtrowaniem po typie/tagu muzyki
  search_url = f"https://archive.org/advancedsearch.php?q=collection%3A({collection_identifier})+AND+{music_type}&fl[]=identifier&rows=50&output=json"
  
  try:
    response = requests.get(search_url, headers=headers, timeout=15)
    if response.status_code != 200:
      print(f"Błąd połączenia z API Archive: {response.status_code}")
      return
    docs = response.json().get("response", {}).get("docs", [])
  except Exception as e:
    print(f"Błąd wyszukiwania: {e}")
    return

  if not docs:
    print(f"Nie znaleziono elementów dla hasła: '{music_type}'.")
    return

  downloaded_count = 0

  for doc in docs:
    if downloaded_count >= target_count:
      break
      
    item_id = doc["identifier"]
    metadata_url = f"https://archive.org/metadata/{item_id}"
    
    try:
      meta_resp = requests.get(metadata_url, headers=headers, timeout=10)
      if meta_resp.status_code != 200:
        continue
        
      data = meta_resp.json()
      files = data.get("files", [])
      server = data.get("server", "")
      dir_path = data.get("dir", "")

      for file in files:
        if downloaded_count >= target_count:
          break
          
        filename = file.get("name", "")
        if filename.endswith(".mp3"):
          download_url = f"https://{server}{dir_path}/{filename}"
          safe_name = f"track_{downloaded_count + 1}.mp3"
          filepath = os.path.join(output_dir, safe_name)
          
          print(f"Pobieranie [{downloaded_count + 1}/{target_count}]: {filename}...")
          
          try:
            file_resp = requests.get(download_url, headers=headers, stream=True, timeout=20)
            if file_resp.status_code != 200:
              continue
              
            total_size = int(file_resp.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(filepath, "wb") as f:
              for chunk in file_resp.iter_content(chunk_size=16384):
                if chunk:
                  f.write(chunk)
                  downloaded_size += len(chunk)
            
            # Weryfikacja integralności pliku
            if total_size > 0 and downloaded_size < total_size:
              print("-> Pobieranie przerwane, plik niekompletny. Usuwam...")
              os.remove(filepath)
              continue
              
            if os.path.getsize(filepath) > 100 * 1024:
              downloaded_count += 1
              time.sleep(0.3)
            else:
              if os.path.exists(filepath):
                os.remove(filepath)
                
          except requests.exceptions.RequestException:
            if os.path.exists(filepath):
              os.remove(filepath)
            continue
              
    except Exception as e:
      continue

  print(f"\nZakończono! Pobrano łącznie {downloaded_count} w pełni poprawnych utworów do folderu: '{output_dir}'.")

if __name__ == "__main__":
  # Możesz zmienić na dowolny gatunek/typ, np. "ambient", "electronic", "classical", "jazz"
  download_from_archive_api(music_type="lofi", target_count=200)