"""Download audio tracks from YouTube using yt-dlp.

This module is intentionally kept small and focused so it can be developed
and tested in isolation from the rest of the pipeline. It supports two ways
of obtaining audio:

* ``download_audio``: download from a known YouTube URL.
* ``search_and_download_audio``: search YouTube by a free-text query (e.g.
  a song or artist name) and download the audio of the best match.

Running this file directly also exposes a small interactive CLI, so a user
can simply type a search phrase and get an audio file back.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional

# Default cap on a single video's duration: videos longer than this are
# skipped entirely rather than downloaded. This avoids accidentally pulling
# in multi-hour live sets/streams (which also balloon preprocessing/training
# time -- a single hour-long file can expand into thousands of training
# segments). Pass ``max_duration_seconds=None`` to disable the limit.
DEFAULT_MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60  # 2 hours

# Default cookies file (Netscape format, e.g. exported via a browser
# extension like "Get cookies.txt") passed to yt-dlp as `cookiefile`. Using
# cookies from a logged-in YouTube session helps avoid bot-detection/
# age-restriction/region-lock errors that an anonymous request would hit.
# Only used if the file actually exists at this path (see
# `_resolve_cookies_file`); harmless/no-op otherwise, so the downloader still
# works out of the box without it.
DEFAULT_COOKIES_FILE = "youtube.com_cookies.txt"


def _resolve_cookies_file(cookies_file: Optional[str]) -> Optional[str]:
    """Resolve the cookies file path to pass to yt-dlp, if it exists.

    Args:
        cookies_file: Path to a Netscape-format cookies file, or ``None``/
            empty string to disable cookies entirely.

    Returns:
        The path as a string if it points to an existing file, otherwise
        ``None`` (so yt-dlp falls back to anonymous requests instead of
        erroring out on a missing file).
    """
    if not cookies_file:
        return None
    return cookies_file if os.path.exists(cookies_file) else None


def _import_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "yt-dlp is required for downloading audio. "
            "Install it with `pip install yt-dlp`."
        ) from exc
    return yt_dlp


def _build_ydl_opts(
    output_dir: str,
    audio_format: str,
    filename: Optional[str] = None,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    cookies_file: Optional[str] = DEFAULT_COOKIES_FILE,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out_template = (
        os.path.join(output_dir, f"{filename}.%(ext)s")
        if filename
        else os.path.join(output_dir, "%(title)s.%(ext)s")
    )

    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    if max_duration_seconds is not None:
        # Belt-and-suspenders duration cap applied at download time (in
        # addition to any upfront filtering done on search results): skips a
        # video if yt-dlp determines its duration exceeds the limit, even if
        # that wasn't known ahead of time (e.g. a direct URL download, or a
        # live/upcoming video whose duration wasn't available during search).
        opts["match_filter"] = _duration_match_filter(max_duration_seconds)

    resolved_cookies_file = _resolve_cookies_file(cookies_file)
    if resolved_cookies_file is not None:
        # Netscape-format cookies file (e.g. exported from a logged-in
        # browser session via a "cookies.txt" extension). Helps yt-dlp avoid
        # YouTube's bot-detection/"Sign in to confirm you're not a bot"
        # errors and access age-restricted content. See
        # https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp
        opts["cookiefile"] = resolved_cookies_file

    return opts


def _duration_match_filter(max_duration_seconds: int):
    """Build a yt-dlp ``match_filter`` rejecting videos longer than the limit.

    A yt-dlp ``match_filter`` is a callable taking ``(info_dict, *,
    incomplete)`` and returning ``None`` to allow the video through, or a
    string (the skip reason) to reject it. Implemented directly (instead of
    relying on yt-dlp's internal ``match_filter_func`` helper/DSL) so it has
    no dependency on yt-dlp internals and is trivially unit-testable/mockable.

    Entries with an unknown duration (``None``) are allowed through -- yt-dlp
    calls the filter once with incomplete info (duration may not be known
    yet) and again once the full info (including duration) is available, so
    rejecting on an unknown duration here could incorrectly block a video
    before its real duration was ever checked.
    """

    def _filter(info_dict: dict, *, incomplete: bool = False) -> Optional[str]:
        duration = info_dict.get("duration")
        if duration is not None and duration > max_duration_seconds:
            return (
                f"video duration {duration}s exceeds max_duration_seconds="
                f"{max_duration_seconds}, skipping .."
            )
        return None

    return _filter


def _resolve_final_path(ydl: Any, info: dict, audio_format: str) -> str:
    base_path = ydl.prepare_filename(info)
    final_path = str(Path(base_path).with_suffix(f".{audio_format}"))

    if not os.path.exists(final_path):
        raise RuntimeError(
            f"Expected downloaded audio file at '{final_path}' but it was not found."
        )

    return final_path


def download_audio(
    url: str,
    output_dir: str,
    audio_format: str = "wav",
    filename: Optional[str] = None,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    cookies_file: Optional[str] = DEFAULT_COOKIES_FILE,
) -> str:
    """Download the audio track of a YouTube video and save it to disk.

    Args:
        url: The YouTube video URL.
        output_dir: Directory where the audio file will be saved. Created if
            it does not already exist.
        audio_format: Target audio codec/container (e.g. "wav", "mp3").
        filename: Optional base filename (without extension). If not
            provided, yt-dlp's video title/id template is used.
        max_duration_seconds: Reject the video if its duration exceeds this
            many seconds (default: 2 hours). Pass ``None`` to disable the
            limit.
        cookies_file: Path to a Netscape-format cookies file (default:
            ``youtube.com_cookies.txt`` in the current working directory),
            passed to yt-dlp as ``cookiefile`` to authenticate requests as a
            logged-in browser session and avoid bot-detection/age/region
            errors. Silently ignored if the file doesn't exist. Pass
            ``None``/``""`` to disable cookies entirely.

    Returns:
        The absolute path to the downloaded audio file.

    Raises:
        ImportError: If the optional ``yt-dlp`` dependency is not installed.
        RuntimeError: If the download finished but no output file was found,
            or the video exceeded ``max_duration_seconds`` and was skipped.
    """
    yt_dlp = _import_yt_dlp()
    ydl_opts = _build_ydl_opts(
        output_dir,
        audio_format,
        filename,
        max_duration_seconds=max_duration_seconds,
        cookies_file=cookies_file,
    )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(
                f"Video at '{url}' was rejected (likely exceeds "
                f"max_duration_seconds={max_duration_seconds})."
            )
        return _resolve_final_path(ydl, info, audio_format)


def search_and_download_audio(
    query: str,
    output_dir: str,
    audio_format: str = "wav",
    filename: Optional[str] = None,
    max_results: int = 1,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    cookies_file: Optional[str] = DEFAULT_COOKIES_FILE,
) -> str:
    """Search YouTube for ``query`` and download the audio of the top match.

    This lets a user provide a free-text phrase (e.g. a song title and/or
    artist name) instead of a direct video URL.

    Args:
        query: Free-text search phrase, e.g. "Artist - Song Title".
        output_dir: Directory where the audio file will be saved. Created if
            it does not already exist.
        audio_format: Target audio codec/container (e.g. "wav", "mp3").
        filename: Optional base filename (without extension) for the
            downloaded result. If not provided, the video's title is used.
        max_results: How many search results yt-dlp should fetch before
            picking the first one. Kept at 1 by default to only download a
            single, best-ranked match.
        max_duration_seconds: Reject the top match if its duration exceeds
            this many seconds (default: 2 hours). Pass ``None`` to disable
            the limit.
        cookies_file: Path to a Netscape-format cookies file (default:
            ``youtube.com_cookies.txt``), see ``download_audio`` for details.
            Silently ignored if the file doesn't exist.

    Returns:
        The absolute path to the downloaded audio file.

    Raises:
        ImportError: If the optional ``yt-dlp`` dependency is not installed.
        ValueError: If ``query`` is empty or blank.
        RuntimeError: If no search results were found, or the download
            finished but no output file was found.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    yt_dlp = _import_yt_dlp()
    ydl_opts = _build_ydl_opts(
        output_dir,
        audio_format,
        filename,
        max_duration_seconds=max_duration_seconds,
        cookies_file=cookies_file,
    )
    search_term = f"ytsearch{max_results}:{query.strip()}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_term, download=True)
        entries = list(info.get("entries") or [])

        if not entries:
            raise RuntimeError(f"No YouTube results found for query: '{query}'")

        return _resolve_final_path(ydl, entries[0], audio_format)


def search_and_download_many(
    query: str,
    output_dir: str,
    count: int = 200,
    audio_format: str = "wav",
    skip_errors: bool = True,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    cookies_file: Optional[str] = DEFAULT_COOKIES_FILE,
) -> List[str]:
    """Search YouTube for ``query`` and download audio for up to ``count`` results.

    This is the bulk counterpart of ``search_and_download_audio``: instead of
    downloading only the single best match, it fetches ``count`` search
    results for the same query and downloads the audio of each one. Useful
    for building a dataset of many tracks from a single keyword/topic.

    Args:
        query: Free-text search phrase, e.g. an artist name or genre.
        output_dir: Directory where audio files will be saved. Created if it
            does not already exist.
        count: Maximum number of videos to download for this query.
        audio_format: Target audio codec/container (e.g. "wav", "mp3").
        skip_errors: If True (default), a failure on an individual video
            (e.g. region-locked, age-restricted, no audio stream) is logged
            and skipped so the rest of the batch can still complete. If
            False, the first error aborts the whole batch.
        max_duration_seconds: Skip any video longer than this many seconds
            (default: 2 hours). Videos are filtered out both upfront (using
            the duration already present in the flat search results, at no
            extra network cost) and again at download time as a fallback for
            entries whose duration wasn't known during search. Pass ``None``
            to disable the limit.
        cookies_file: Path to a Netscape-format cookies file (default:
            ``youtube.com_cookies.txt``), see ``download_audio`` for details.
            Silently ignored if the file doesn't exist. Used for both the
            flat search step and every individual video download.

    Returns:
        A list of paths to the successfully downloaded audio files. May have
        fewer than ``count`` entries if some results failed and
        ``skip_errors`` is True, or if fewer than ``count`` results exist.

    Raises:
        ImportError: If the optional ``yt-dlp`` dependency is not installed.
        ValueError: If ``query`` is empty/blank or ``count`` is not positive.
        RuntimeError: If no search results were found at all.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")
    if count <= 0:
        raise ValueError("count must be a positive integer.")

    yt_dlp = _import_yt_dlp()
    ydl_opts = _build_ydl_opts(
        output_dir,
        audio_format,
        max_duration_seconds=max_duration_seconds,
        cookies_file=cookies_file,
    )
    # Over-fetch search results: ytsearchN sometimes returns non-video entries
    # (e.g. a channel or "YoutubeTab" page instead of an actual video), which
    # must be filtered out before counting towards the requested ``count``.
    # Requesting extra candidates up front avoids a second network round-trip
    # when some entries turn out to be unusable.
    fetch_count = max(count * 2, count + 10)
    search_term = f"ytsearch{fetch_count}:{query.strip()}"

    # IMPORTANT: use a *flat* extraction for the search step. Without
    # `extract_flat`, yt-dlp eagerly resolves every search result in full,
    # including recursively expanding any channel/playlist entry into all of
    # its individual videos *before even returning the search results list*.
    # That defeats any later filtering (the expanded videos look like normal
    # "Youtube" entries) and can hang for a long time or silently download
    # far more than `count` tracks. `extract_flat="in_playlist"` keeps each
    # search result as a lightweight reference (with an accurate `ie_key`)
    # without expanding containers, so channels/playlists can be filtered out
    # below before anything is downloaded.
    search_opts = dict(ydl_opts)
    search_opts["extract_flat"] = "in_playlist"

    downloaded_paths: List[str] = []

    with yt_dlp.YoutubeDL(search_opts) as search_ydl:
        # First, resolve the search results WITHOUT downloading, so a single
        # broken entry can't abort metadata extraction for the whole batch.
        search_info = search_ydl.extract_info(search_term, download=False)
        all_entries = list(search_info.get("entries") or [])

    # Keep only entries that look like single, directly downloadable videos.
    # Entries whose ie_key/_type indicate a channel, playlist, or other
    # container (e.g. "YoutubeTab") are skipped here -- passing such a URL to
    # extract_info(download=True) would recursively download many/all videos
    # from that channel/playlist, blowing past the requested ``count`` with
    # no predictable stopping point.
    video_entries = [
        entry
        for entry in all_entries
        if entry and entry.get("ie_key", "Youtube") == "Youtube"
    ]

    # Also filter out videos that are already known to be too long, using the
    # `duration` field yt-dlp's flat search already provides -- this skips
    # them for free, without spending any time/bandwidth resolving or
    # downloading them. Entries with an unknown duration at this stage (e.g.
    # `None`) are kept; they are still protected by the `match_filter` set on
    # `ydl_opts` below, which re-checks duration once it's known at download
    # time.
    if max_duration_seconds is not None:
        video_entries = [
            entry
            for entry in video_entries
            if entry.get("duration") is None
            or entry.get("duration") <= max_duration_seconds
        ]

    if not video_entries:
        raise RuntimeError(f"No YouTube results found for query: '{query}'")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for entry in video_entries:
            # Hard cap: stop as soon as we have exactly `count` successful
            # downloads, regardless of how many candidate entries remain.
            if len(downloaded_paths) >= count:
                break

            video_url = entry.get("webpage_url") or entry.get("url")
            try:
                video_info = ydl.extract_info(video_url, download=True)
                if video_info is None:
                    # yt-dlp's match_filter (e.g. duration cap) rejected this
                    # video at download time -- it returns None instead of
                    # raising. Treat it the same as a skipped/failed entry.
                    print(
                        f"[WARN] Skipping '{video_url}': rejected by match_filter "
                        f"(likely exceeds max_duration_seconds={max_duration_seconds})"
                    )
                    continue
                downloaded_paths.append(
                    _resolve_final_path(ydl, video_info, audio_format)
                )
            except Exception as exc:  # noqa: BLE001 - intentionally broad for batch robustness
                if not skip_errors:
                    raise
                print(f"[WARN] Skipping '{video_url}': {exc}")

    return downloaded_paths


def main(argv: Optional[list] = None) -> None:
    """Small interactive CLI: type a search phrase, get an audio file back."""
    parser = argparse.ArgumentParser(
        description="Search YouTube by keyword and download the audio of the top match."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search phrase (e.g. song or artist name). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save the downloaded audio file (default: data/raw).",
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default="wav",
        help="Target audio format, e.g. wav, mp3 (default: wav).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of videos to download for the query (default: 1, the best match). "
        "Use e.g. --count 200 to download many tracks for the same query.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=DEFAULT_MAX_VIDEO_DURATION_SECONDS,
        help="Skip videos longer than this many seconds (default: 7200 = 2 hours). "
        "Use 0 or a negative value to disable the limit.",
    )
    parser.add_argument(
        "--cookies-file",
        default=DEFAULT_COOKIES_FILE,
        help="Path to a Netscape-format cookies file passed to yt-dlp, used to "
        f"authenticate as a logged-in browser session (default: {DEFAULT_COOKIES_FILE}, "
        "silently ignored if it doesn't exist). Pass an empty string to disable cookies.",
    )
    args = parser.parse_args(argv)

    query = args.query or input("Wpisz słowo/frazę do wyszukania na YouTube: ").strip()
    if not query:
        raise SystemExit("Nie podano frazy wyszukiwania.")

    max_duration_seconds = args.max_duration if args.max_duration and args.max_duration > 0 else None

    if args.count == 1:
        path = search_and_download_audio(
            query,
            args.output_dir,
            args.audio_format,
            max_duration_seconds=max_duration_seconds,
            cookies_file=args.cookies_file,
        )
        print(f"Pobrano: {path}")
    else:
        paths = search_and_download_many(
            query,
            args.output_dir,
            count=args.count,
            audio_format=args.audio_format,
            max_duration_seconds=max_duration_seconds,
            cookies_file=args.cookies_file,
        )
        print(f"Pobrano {len(paths)}/{args.count} plik(ów) dla zapytania '{query}'.")


if __name__ == "__main__":
    main()
