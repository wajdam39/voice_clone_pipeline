"""Download CC-licensed audio tracks from Internet Archive (archive.org).

This module is intentionally kept small and focused so it can be developed
and tested in isolation from the rest of the pipeline. It uses the Internet
Archive Advanced Search API (no API key required) to find audio items whose
license permits reuse (e.g. Creative Commons), then downloads the audio
files directly from archive.org's download servers.

Supported ways of obtaining audio:

* ``download_audio``: download audio for a known Internet Archive identifier
  (e.g. ``some-collection-item``).
* ``search_and_download_audio``: search Internet Archive by a free-text query
  filtered to Creative Commons-licensed audio and download the top match.
* ``search_and_download_many``: bulk download up to ``count`` CC-licensed
  tracks for a single query — useful for building a dataset from a keyword.

Running this file directly also exposes a small interactive CLI.

IMPORTANT / LEGAL NOTE: Internet Archive hosts items under many different
licenses. This module filters results to Creative Commons / open licenses by
default, but it is the user's responsibility to verify the license of each
item before reuse (e.g. training a voice-cloning model). Only download and
use content you have the rights to.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import urllib.request
import urllib.error

# Default cap on a single audio item's duration (seconds). Items longer than
# this are skipped entirely to avoid pulling in multi-hour recordings that
# would balloon preprocessing/training time. Pass ``None`` to disable.
DEFAULT_MAX_DURATION_SECONDS = 2 * 60 * 60  # 2 hours

# Default output directory for downloaded audio.
DEFAULT_OUTPUT_DIR = "cc_audio_downloads"

# Default target audio format (Internet Archive serves original files, usually
# MP3; we keep the original extension and let preprocessing handle conversion).
DEFAULT_AUDIO_FORMAT = "mp3"

# Number of search results to fetch per page from the Advanced Search API.
DEFAULT_ROWS_PER_PAGE = 50

# Creative Commons / open license URL patterns used to filter search results.
# An item matches if its ``licenseurl`` field contains any of these substrings.
DEFAULT_CC_LICENSE_URLS = (
    "creativecommons.org/licenses/by/4.0",
    "creativecommons.org/licenses/by-sa/4.0",
    "creativecommons.org/licenses/by-nc/4.0",
    "creativecommons.org/licenses/by-nc-sa/4.0",
    "creativecommons.org/licenses/by-nd/4.0",
    "creativecommons.org/licenses/by-nc-nd/4.0",
    "creativecommons.org/publicdomain/zero/1.0",  # CC0
    "creativecommons.org/publicdomain/mark/1.0",   # PDM
)

# Base URL for the Internet Archive Advanced Search API.
IA_ADVANCED_SEARCH_URL = "https://archive.org/advancedsearch.php"

# Base URL for the Internet Archive metadata API.
IA_METADATA_URL = "https://archive.org/metadata/"

# Base URL for downloading files from Internet Archive.
IA_DOWNLOAD_URL = "https://archive.org/download/"


def _build_ia_query(
    query: str,
    cc_license_urls: Optional[Union[str, List[str], tuple]] = None,
) -> str:
    """Build an Internet Archive advanced search query string.

    The query is restricted to audio items (mediatype:audio) and, by default,
    to items whose license URL matches one of the Creative Commons / open
    license patterns. Additional free-text terms from ``query`` are ANDed in.

    Args:
        query: Free-text search phrase, e.g. "voice narration", "poetry",
            "language lesson".
        cc_license_urls: License URL substring(s) to require, or None/empty to
            disable the CC filter (not recommended — downloads may then include
            copyrighted material).

    Returns:
        A query string suitable for the ``q`` parameter of the Advanced Search
        API.
    """
    clauses: List[str] = ["mediatype:audio"]

    license_urls = (
        list(cc_license_urls) if cc_license_urls else list(DEFAULT_CC_LICENSE_URLS)
    )
    if license_urls:
        license_clause = " OR ".join(
            f'licenseurl:"{url}"' for url in license_urls
        )
        clauses.append(f"({license_clause})")

    if query and query.strip():
        clauses.append(f"({query.strip()})")

    return " AND ".join(clauses)


def _ia_search(
    query: str,
    rows: int = DEFAULT_ROWS_PER_PAGE,
    page: int = 1,
    cc_license_urls: Optional[Union[str, List[str], tuple]] = None,
) -> dict:
    """Call the Internet Archive Advanced Search API and return parsed JSON.

    Args:
        query: Free-text search phrase.
        rows: Maximum number of results to return (page size).
        page: 1-based page number for pagination.
        cc_license_urls: Optional license URL filter override.

    Returns:
        Parsed JSON response from the Advanced Search API.

    Raises:
        RuntimeError: If the request fails or returns a non-200 status.
    """
    ia_query = _build_ia_query(query, cc_license_urls)
    start = (page - 1) * rows

    params = {
        "q": ia_query,
        "fl[]": [
            "identifier",
            "title",
            "creator",
            "date",
            "licenseurl",
            "duration",
            "format",
            "mediatype",
            "description",
        ],
        "rows": rows,
        "page": page,
        "output": "json",
    }

    qs = "&".join(
        f"{k}={v}" if not isinstance(v, list) else "&".join(f"{k}={item}" for item in v)
        for k, v in params.items()
    )
    url = f"{IA_ADVANCED_SEARCH_URL}?{qs}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "voice-clone-pipeline/1.0 (contact: example@example.com)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Internet Archive API returned status {resp.status}"
                )
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to query Internet Archive: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse Internet Archive API response: {exc}"
        ) from exc

    return data


def _ia_metadata(identifier: str) -> dict:
    """Fetch the metadata JSON for a given Internet Archive identifier.

    Args:
        identifier: The Internet Archive item identifier (e.g. ``some-item``).

    Returns:
        Parsed metadata JSON dict containing ``files`` and other metadata.

    Raises:
        RuntimeError: If the request fails or the item is not found.
    """
    url = f"{IA_METADATA_URL}{identifier}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "voice-clone-pipeline/1.0 (contact: example@example.com)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Internet Archive metadata API returned status {resp.status} "
                    f"for identifier '{identifier}'"
                )
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"Internet Archive item not found: '{identifier}'"
            ) from exc
        raise RuntimeError(
            f"Failed to fetch metadata for '{identifier}': {exc}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to reach Internet Archive metadata API: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse metadata JSON for '{identifier}': {exc}"
        ) from exc


def _select_audio_file(
    files: List[dict], preferred_formats: Optional[List[str]] = None
) -> Optional[dict]:
    """Pick the best audio file from an item's ``files`` list.

    Prefers common lossy/lossless audio formats (MP3, FLAC, WAV, OGG, M4A,
    AAC) over other file types (text, images, metadata). Within audio files,
    prefers the formats listed in ``preferred_formats`` in order.

    Args:
        files: List of file dicts from the item metadata.
        preferred_formats: Ordered list of preferred extensions (e.g.
            ["mp3", "flac", "wav"]). Defaults to a sensible order.

    Returns:
        The selected file dict, or None if no audio file is found.
    """
    if preferred_formats is None:
        preferred_formats = ["mp3", "flac", "wav", "ogg", "m4a", "aac"]

    audio_extensions = set(preferred_formats)

    audio_files = [
        f
        for f in files
        if f.get("format")
        and f["format"].lower() in {ext.upper() for ext in audio_extensions}
        or any(f.get("name", "").lower().endswith(f".{ext}") for ext in audio_extensions)
    ]

    if not audio_files:
        return None

    def sort_key(f: dict) -> tuple:
        name = f.get("name", "").lower()
        try:
            pref_idx = preferred_formats.index(
                next(ext for ext in preferred_formats if name.endswith(f".{ext}"))
            )
        except StopIteration:
            pref_idx = len(preferred_formats)
        try:
            size = int(f.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        return (pref_idx, -size)

    return sorted(audio_files, key=sort_key)[0]


def _download_file(url: str, output_path: str) -> str:
    """Download a single file from ``url`` to ``output_path``.

    Args:
        url: The direct download URL.
        output_path: Local path to save the file to.

    Returns:
        The local path to the downloaded file.

    Raises:
        RuntimeError: If the download fails.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "voice-clone-pipeline/1.0 (contact: example@example.com)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(output_path, "wb") as out_file:
            if resp.status != 200:
                raise RuntimeError(f"Download failed with status {resp.status}")
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download '{url}': {exc}") from exc

    return output_path


def download_audio(
    identifier: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    filename: Optional[str] = None,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_DURATION_SECONDS,
    preferred_formats: Optional[List[str]] = None,
) -> str:
    """Download audio for an Internet Archive identifier.

    Fetches the item metadata, selects the best audio file, and downloads it
    to ``output_dir``.

    Args:
        identifier: Internet Archive item identifier (e.g. ``some-item``).
        output_dir: Directory to save the audio file into. Created if needed.
        audio_format: Preferred audio format extension (e.g. ``mp3``, ``wav``,
            ``flac``). Used as a hint when selecting among multiple audio
            files; the original file is downloaded as-is.
        filename: Optional base filename (without extension). If not provided,
            the original filename from archive.org is used.
        max_duration_seconds: Skip the item if its duration exceeds this many
            seconds (default: 2 hours). Pass ``None`` to disable the limit.
        preferred_formats: Ordered list of preferred audio extensions for
            selecting among multiple files on the item.

    Returns:
        The absolute path to the downloaded audio file.

    Raises:
        RuntimeError: If the item has no downloadable audio file, the download
            fails, or the item exceeds ``max_duration_seconds``.
    """
    metadata = _ia_metadata(identifier)
    files = metadata.get("files", []) or []
    item_metadata = metadata.get("metadata", {}) or {}

    duration = item_metadata.get("duration")
    if duration is not None:
        try:
            duration_val = float(duration)
        except (TypeError, ValueError):
            duration_val = None
        if duration_val is not None and max_duration_seconds is not None:
            if duration_val > max_duration_seconds:
                raise RuntimeError(
                    f"Item '{identifier}' duration {duration_val}s exceeds "
                    f"max_duration_seconds={max_duration_seconds}, skipping."
                )

    selected = _select_audio_file(files, preferred_formats=preferred_formats)
    if selected is None:
        raise RuntimeError(
            f"No downloadable audio file found for item '{identifier}'."
        )

    file_name = selected.get("name", "")
    if not file_name:
        raise RuntimeError(
            f"Selected file for '{identifier}' has no name."
        )

    if filename:
        ext = Path(file_name).suffix or f".{audio_format}"
        out_name = f"{filename}{ext}"
    else:
        out_name = file_name

    output_path = str(Path(output_dir) / out_name)

    download_url = f"{IA_DOWNLOAD_URL}{identifier}/{file_name}"
    _download_file(download_url, output_path)

    return output_path


def search_and_download_audio(
    query: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    filename: Optional[str] = None,
    max_results: int = 1,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_DURATION_SECONDS,
    cc_license_urls: Optional[Union[str, List[str], tuple]] = None,
    preferred_formats: Optional[List[str]] = None,
) -> str:
    """Search Internet Archive for ``query`` and download the top CC-licensed match.

    Args:
        query: Free-text search phrase, e.g. "voice narration", "poetry reading".
        output_dir: Directory to save the audio file into.
        audio_format: Preferred audio format hint (see ``download_audio``).
        filename: Optional base filename for the downloaded result.
        max_results: How many search results to consider before picking the
            first one that has a downloadable audio file.
        max_duration_seconds: Skip items longer than this many seconds
            (default: 2 hours). Pass ``None`` to disable.
        cc_license_urls: Optional license URL filter override; defaults to
            Common Creative / open licenses.
        preferred_formats: Ordered list of preferred audio extensions.

    Returns:
        The absolute path to the downloaded audio file.

    Raises:
        ValueError: If ``query`` is empty or blank.
        RuntimeError: If no results are found, or no result has a downloadable
            audio file.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    search_data = _ia_search(
        query,
        rows=max(max_results * 3, max_results + 5),
        cc_license_urls=cc_license_urls,
    )

    response = search_data.get("response", {}) or {}
    docs = response.get("docs", []) or []

    if not docs:
        raise RuntimeError(
            f"No Internet Archive results found for query: '{query}'"
        )

    last_error: Optional[Exception] = None
    for doc in docs[:max_results]:
        identifier = doc.get("identifier")
        if not identifier:
            continue
        try:
            return download_audio(
                identifier,
                output_dir=output_dir,
                audio_format=audio_format,
                filename=filename,
                max_duration_seconds=max_duration_seconds,
                preferred_formats=preferred_formats,
            )
        except Exception as exc:  # noqa: BLE001 - try next result on failure
            last_error = exc
            continue

    if last_error:
        raise RuntimeError(
            f"Found results for '{query}' but none could be downloaded: {last_error}"
        )
    raise RuntimeError(
        f"No downloadable audio files found for query: '{query}'"
    )


def search_and_download_many(
    query: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    count: int = 200,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    skip_errors: bool = True,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_DURATION_SECONDS,
    cc_license_urls: Optional[Union[str, List[str], tuple]] = None,
    preferred_formats: Optional[List[str]] = None,
) -> List[str]:
    """Search Internet Archive for ``query`` and download up to ``count`` tracks.

    This is the bulk counterpart of ``search_and_download_audio``: it fetches
    multiple search results for the same query and downloads the audio of each
    one that has a downloadable file and a CC-compatible license.

    Args:
        query: Free-text search phrase, e.g. an artist name, genre, or topic.
        output_dir: Directory where audio files will be saved.
        count: Maximum number of tracks to download.
        audio_format: Preferred audio format hint (see ``download_audio``).
        skip_errors: If True (default), a failure on an individual item is
            logged and skipped so the rest of the batch can complete. If
            False, the first error aborts the whole batch.
        max_duration_seconds: Skip items longer than this many seconds
            (default: 2 hours). Pass ``None`` to disable.
        cc_license_urls: Optional license URL filter override; defaults to
            Common Creative / open licenses.
        preferred_formats: Ordered list of preferred audio extensions.

    Returns:
        A list of paths to the successfully downloaded audio files. May have
        fewer than ``count`` entries if some items failed and ``skip_errors``
        is True, or if fewer than ``count`` results exist.

    Raises:
        ValueError: If ``query`` is empty/blank or ``count`` is not positive.
        RuntimeError: If no search results were found at all.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")
    if count <= 0:
        raise ValueError("count must be a positive integer.")

    search_data = _ia_search(
        query,
        rows=max(count * 3, count + 10),
        cc_license_urls=cc_license_urls,
    )

    response = search_data.get("response", {}) or {}
    docs = response.get("docs", []) or []

    if not docs:
        raise RuntimeError(
            f"No Internet Archive results found for query: '{query}'"
        )

    downloaded_paths: List[str] = []

    for doc in docs:
        if len(downloaded_paths) >= count:
            break

        identifier = doc.get("identifier")
        if not identifier:
            continue

        try:
            path = download_audio(
                identifier,
                output_dir=output_dir,
                audio_format=audio_format,
                max_duration_seconds=max_duration_seconds,
                preferred_formats=preferred_formats,
            )
            downloaded_paths.append(path)
        except Exception as exc:  # noqa: BLE001 - intentionally broad for batch robustness
            if not skip_errors:
                raise
            print(f"[WARN] Skipping '{identifier}': {exc}")

    return downloaded_paths


def main(argv: Optional[list] = None) -> None:
    """Small interactive CLI: type a search phrase, get CC audio back."""
    parser = argparse.ArgumentParser(
        description=(
            "Search Internet Archive for Creative Commons-licensed audio "
            "and download it. Only download content you have the rights to use."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search phrase (e.g. voice narration, poetry, language lesson). "
        "If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded audio files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default=DEFAULT_AUDIO_FORMAT,
        help="Preferred audio format hint, e.g. mp3, wav, flac (default: mp3).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of tracks to download for the query (default: 1, the best match). "
        "Use e.g. --count 200 to download many tracks.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help="Skip items longer than this many seconds (default: 7200 = 2 hours). "
        "Use 0 or a negative value to disable the limit.",
    )
    parser.add_argument(
        "--cc-license-urls",
        default=",".join(DEFAULT_CC_LICENSE_URLS),
        help="Comma-separated Creative Commons license URL substrings to require "
        f"(default: {','.join(DEFAULT_CC_LICENSE_URLS)!r}). Pass an empty string "
        "to disable the CC filter (not recommended).",
    )
    parser.add_argument(
        "--preferred-formats",
        default="mp3,flac,wav,ogg,m4a,aac",
        help="Comma-separated preferred audio extensions (default: mp3,flac,wav,ogg,m4a,aac).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the whole batch on the first error instead of skipping it.",
    )
    args = parser.parse_args(argv)

    query = args.query or input("Wpisz frazę do wyszukania na Internet Archive: ").strip()
    if not query:
        raise SystemExit("Nie podano frazy wyszukiwania.")

    max_duration_seconds = (
        args.max_duration if args.max_duration and args.max_duration > 0 else None
    )
    cc_license_urls = (
        [c.strip() for c in args.cc_license_urls.split(",") if c.strip()]
        if args.cc_license_urls
        else None
    )
    preferred_formats = (
        [c.strip() for c in args.preferred_formats.split(",") if c.strip()]
        if args.preferred_formats
        else None
    )

    if args.count == 1:
        path = search_and_download_audio(
            query,
            args.output_dir,
            args.audio_format,
            max_duration_seconds=max_duration_seconds,
            cc_license_urls=cc_license_urls,
            preferred_formats=preferred_formats,
        )
        print(f"Pobrano: {path}")
    else:
        paths = search_and_download_many(
            query,
            args.output_dir,
            count=args.count,
            audio_format=args.audio_format,
            skip_errors=not args.fail_fast,
            max_duration_seconds=max_duration_seconds,
            cc_license_urls=cc_license_urls,
            preferred_formats=preferred_formats,
        )
        print(f"Pobrano {len(paths)}/{args.count} plik(ów) dla zapytania '{query}'.")


if __name__ == "__main__":
    main()