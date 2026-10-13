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
from typing import Any, List, Optional, Union

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

# YouTube now requires solving a JavaScript "challenge" (EJS, External JS
# Scripts) for most requests; without a working JS runtime, downloads can
# fail/get blocked entirely -- this is especially common from datacenter IPs
# (e.g. AWS EC2), which YouTube flags much more aggressively than residential
# IPs. yt-dlp solves this by running challenge-solver scripts in an external
# JS runtime (deno/node/quickjs/bun), optionally fetched on demand from
# GitHub/npm ("remote components"). See
# https://github.com/yt-dlp/yt-dlp/wiki/ejs
#
# "node" is used as the default runtime (rather than yt-dlp's own default of
# "deno") because Node.js is far more likely to already be installed/easily
# installable on a typical EC2 instance/CI image than Deno. Override via
# `js_runtime` if you do have deno/quickjs/bun available instead.
DEFAULT_JS_RUNTIME = "node"

# Allow yt-dlp to download the EJS challenge-solver scripts from GitHub on
# demand (cached locally afterwards, see yt-dlp's --cache-dir). This is what
# lets the "node"/"deno" runtime above actually solve YouTube's JS challenge
# instead of merely being enabled with nothing to run. Set to an empty tuple
# to disable (e.g. if GitHub is not reachable from your network).
DEFAULT_REMOTE_COMPONENTS = ("ejs:github",)

# Random delay (in seconds, uniformly between these two bounds) that yt-dlp
# waits before each download. Spacing out requests like this makes traffic
# look less like an automated scraper and reduces the odds of triggering
# YouTube's rate-limiting/bot-detection -- especially important when
# downloading many videos back-to-back (e.g. search_and_download_many) from
# a single datacenter IP (e.g. EC2), which is scrutinized far more than a
# residential IP would be. Set either to ``None`` to disable sleeping.
DEFAULT_SLEEP_INTERVAL_SECONDS = 5
DEFAULT_MAX_SLEEP_INTERVAL_SECONDS = 15

# Which YouTube "player client(s)" yt-dlp impersonates when requesting
# playback data, passed via extractor_args -> youtube -> player_client. The
# default web/mweb clients are the ones most tightly tied to the JS
# challenge/PO Token requirements and the first to get throttled/blocked on
# datacenter IPs (e.g. EC2). "ios" and "tv_embedded" are mobile/TV clients
# that historically don't require solving YouTube's JS challenge at all and
# are far less likely to be blocked, making them a good complement to (not
# a replacement for) the cookies/js_runtime/remote_components settings
# above. See https://github.com/yt-dlp/yt-dlp/wiki/Extractors#youtube
DEFAULT_PLAYER_CLIENTS = ("ios", "tv_embedded")


def _resolve_js_runtimes(
    js_runtime: Optional[str],
) -> Optional[dict]:
    """Build the ``js_runtimes`` dict yt-dlp's Python API expects.

    Args:
        js_runtime: Name of a single JS runtime to enable (e.g. ``"node"``,
            ``"deno"``, ``"quickjs"``, ``"bun"``), optionally suffixed with
            ``:PATH`` to point at a non-PATH executable (e.g.
            ``"node:/usr/local/bin/node"``), mirroring yt-dlp's ``--js-runtimes``
            CLI syntax. Pass ``None``/``""`` to fall back to yt-dlp's own
            default (``deno``, auto-enabled).

    Returns:
        A ``{runtime_name: {"path": path_or_None}}`` dict as expected by
        ``YoutubeDL(params={"js_runtimes": ...})``, or ``None`` to let yt-dlp
        use its built-in default.
    """
    if not js_runtime:
        return None
    name, _, path = js_runtime.partition(":")
    return {name.strip().lower(): {"path": path.strip() or None}}


def _resolve_remote_components(
    remote_components: Optional[Union[str, List[str], tuple]],
) -> Optional[List[str]]:
    """Normalize ``remote_components`` into the list yt-dlp's API expects.

    Args:
        remote_components: A single component string (e.g. ``"ejs:github"``),
            an iterable of such strings, or ``None``/empty to disable remote
            component downloads entirely.

    Returns:
        A list of component strings, or ``None`` if none were requested.
    """
    if not remote_components:
        return None
    if isinstance(remote_components, str):
        return [remote_components]
    return list(remote_components)


def _resolve_player_clients(
    player_clients: Optional[Union[str, List[str], tuple]],
) -> Optional[dict]:
    """Build the ``extractor_args`` dict yt-dlp's Python API expects for
    overriding the YouTube player client(s).

    Args:
        player_clients: A single client name (e.g. ``"ios"``), an iterable
            of client names (e.g. ``["ios", "tv_embedded"]``), or
            ``None``/empty to disable (falls back to yt-dlp's own default
            client selection).

    Returns:
        An ``{"youtube": {"player_client": [...]}}`` dict as expected by
        ``YoutubeDL(params={"extractor_args": ...})``, or ``None``.
    """
    if not player_clients:
        return None
    clients = [player_clients] if isinstance(player_clients, str) else list(player_clients)
    return {"youtube": {"player_client": clients}}


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
    js_runtime: Optional[str] = DEFAULT_JS_RUNTIME,
    remote_components: Optional[Union[str, List[str], tuple]] = DEFAULT_REMOTE_COMPONENTS,
    sleep_interval: Optional[float] = DEFAULT_SLEEP_INTERVAL_SECONDS,
    max_sleep_interval: Optional[float] = DEFAULT_MAX_SLEEP_INTERVAL_SECONDS,
    player_clients: Optional[Union[str, List[str], tuple]] = DEFAULT_PLAYER_CLIENTS,
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

    resolved_js_runtimes = _resolve_js_runtimes(js_runtime)
    if resolved_js_runtimes is not None:
        # Enables an external JS runtime (e.g. Node.js) so yt-dlp can solve
        # YouTube's JS challenge (EJS). Without this, requests from
        # datacenter/cloud IPs (e.g. EC2) are very likely to be blocked.
        opts["js_runtimes"] = resolved_js_runtimes

    resolved_remote_components = _resolve_remote_components(remote_components)
    if resolved_remote_components is not None:
        # Lets yt-dlp fetch the EJS challenge-solver scripts themselves (the
        # code the JS runtime above actually executes) on demand from GitHub.
        opts["remote_components"] = resolved_remote_components

    if sleep_interval is not None:
        # Minimum delay (seconds) yt-dlp waits before each download. Combined
        # with max_sleep_interval below, yt-dlp sleeps a random duration
        # uniformly between the two, making request timing look less
        # bot-like. (yt-dlp requires sleep_interval to be set whenever
        # max_sleep_interval is -- it errors with a bare max_sleep_interval.)
        opts["sleep_interval"] = sleep_interval
        if max_sleep_interval is not None:
            opts["max_sleep_interval"] = max_sleep_interval

    resolved_player_clients = _resolve_player_clients(player_clients)
    if resolved_player_clients is not None:
        # Forces yt-dlp's YouTube extractor to use specific player client(s)
        # (e.g. "ios", "tv_embedded") instead of its own default selection.
        # Mobile/TV clients historically don't require solving YouTube's JS
        # challenge and are less likely to be rate-limited/blocked from
        # datacenter IPs (e.g. EC2), complementing cookies/js_runtime above.
        opts["extractor_args"] = resolved_player_clients

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
    js_runtime: Optional[str] = DEFAULT_JS_RUNTIME,
    remote_components: Optional[Union[str, List[str], tuple]] = DEFAULT_REMOTE_COMPONENTS,
    sleep_interval: Optional[float] = DEFAULT_SLEEP_INTERVAL_SECONDS,
    max_sleep_interval: Optional[float] = DEFAULT_MAX_SLEEP_INTERVAL_SECONDS,
    player_clients: Optional[Union[str, List[str], tuple]] = DEFAULT_PLAYER_CLIENTS,
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
        js_runtime: External JavaScript runtime used by yt-dlp to solve
            YouTube's JS challenge (EJS), as ``"name"`` or ``"name:path"``
            (default: ``"node"``). Without a working JS runtime, downloads
            from datacenter/cloud IPs (e.g. AWS EC2) are very likely to be
            blocked. Pass ``None``/``""`` to fall back to yt-dlp's own
            default (``deno``).
        remote_components: Remote component(s) yt-dlp is allowed to fetch on
            demand, e.g. ``"ejs:github"`` (default) to download the EJS
            challenge-solver scripts from GitHub so ``js_runtime`` has
            something to execute. Pass ``None``/empty to disable.
        sleep_interval: Minimum number of seconds yt-dlp sleeps before each
            download (default: 5). When ``max_sleep_interval`` is also set,
            the actual sleep is a random duration uniformly chosen between
            the two, which helps avoid YouTube's rate-limiting/bot-detection
            when making many requests from the same IP. Pass ``None`` to
            disable sleeping entirely.
        max_sleep_interval: Maximum number of seconds yt-dlp sleeps before
            each download (default: 15). Only used if ``sleep_interval`` is
            also set.
        player_clients: YouTube player client(s) yt-dlp should impersonate
            (default: ``("ios", "tv_embedded")``), passed via
            ``extractor_args -> youtube -> player_client``. Mobile/TV
            clients historically don't require solving YouTube's JS
            challenge and are less likely to be blocked from datacenter
            IPs. Pass ``None``/empty to fall back to yt-dlp's own default
            client selection.

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
        js_runtime=js_runtime,
        remote_components=remote_components,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
        player_clients=player_clients,
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
    js_runtime: Optional[str] = DEFAULT_JS_RUNTIME,
    remote_components: Optional[Union[str, List[str], tuple]] = DEFAULT_REMOTE_COMPONENTS,
    sleep_interval: Optional[float] = DEFAULT_SLEEP_INTERVAL_SECONDS,
    max_sleep_interval: Optional[float] = DEFAULT_MAX_SLEEP_INTERVAL_SECONDS,
    player_clients: Optional[Union[str, List[str], tuple]] = DEFAULT_PLAYER_CLIENTS,
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
        js_runtime: External JS runtime used to solve YouTube's JS challenge
            (default: ``"node"``), see ``download_audio`` for details.
        remote_components: Remote component(s) yt-dlp may fetch on demand
            (default: ``"ejs:github"``), see ``download_audio`` for details.
        sleep_interval: Minimum seconds yt-dlp sleeps before each download
            (default: 5), see ``download_audio`` for details.
        max_sleep_interval: Maximum seconds yt-dlp sleeps before each
            download (default: 15), see ``download_audio`` for details.
        player_clients: YouTube player client(s) to impersonate (default:
            ``("ios", "tv_embedded")``), see ``download_audio`` for details.

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
        js_runtime=js_runtime,
        remote_components=remote_components,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
        player_clients=player_clients,
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
    js_runtime: Optional[str] = DEFAULT_JS_RUNTIME,
    remote_components: Optional[Union[str, List[str], tuple]] = DEFAULT_REMOTE_COMPONENTS,
    sleep_interval: Optional[float] = DEFAULT_SLEEP_INTERVAL_SECONDS,
    max_sleep_interval: Optional[float] = DEFAULT_MAX_SLEEP_INTERVAL_SECONDS,
    player_clients: Optional[Union[str, List[str], tuple]] = DEFAULT_PLAYER_CLIENTS,
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
        js_runtime: External JS runtime used to solve YouTube's JS challenge
            (default: ``"node"``), see ``download_audio`` for details.
        remote_components: Remote component(s) yt-dlp may fetch on demand
            (default: ``"ejs:github"``), see ``download_audio`` for details.
        sleep_interval: Minimum seconds yt-dlp sleeps before each download
            (default: 5), see ``download_audio`` for details. Especially
            relevant here, since this function issues many requests
            back-to-back -- without spacing them out, YouTube is much more
            likely to rate-limit or block the (likely datacenter) IP making
            the requests.
        max_sleep_interval: Maximum seconds yt-dlp sleeps before each
            download (default: 15), see ``download_audio`` for details.
        player_clients: YouTube player client(s) to impersonate (default:
            ``("ios", "tv_embedded")``), see ``download_audio`` for details.

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
        js_runtime=js_runtime,
        remote_components=remote_components,
        player_clients=player_clients,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
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
    parser.add_argument(
        "--js-runtime",
        default=DEFAULT_JS_RUNTIME,
        help="External JavaScript runtime used by yt-dlp to solve YouTube's JS "
        f"challenge (default: {DEFAULT_JS_RUNTIME!r}; mirrors yt-dlp's --js-runtimes "
        "syntax, e.g. 'node', 'deno', 'node:/path/to/node'). Without a working JS "
        "runtime, downloads from datacenter/cloud IPs (e.g. EC2) are very likely to "
        "be blocked. Pass an empty string to fall back to yt-dlp's own default (deno).",
    )
    parser.add_argument(
        "--remote-components",
        default=",".join(DEFAULT_REMOTE_COMPONENTS),
        help="Comma-separated remote component(s) yt-dlp may fetch on demand "
        f"(default: {','.join(DEFAULT_REMOTE_COMPONENTS)!r}) -- lets the JS runtime "
        "above actually solve YouTube's JS challenge by downloading the EJS "
        "challenge-solver scripts from GitHub. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=DEFAULT_SLEEP_INTERVAL_SECONDS,
        help="Minimum number of seconds yt-dlp sleeps before each download "
        f"(default: {DEFAULT_SLEEP_INTERVAL_SECONDS}). Combined with "
        "--max-sleep-interval, the actual sleep is a random duration between "
        "the two, reducing the odds of YouTube's rate-limiting/bot-detection. "
        "Pass a negative value to disable sleeping entirely.",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=DEFAULT_MAX_SLEEP_INTERVAL_SECONDS,
        help="Maximum number of seconds yt-dlp sleeps before each download "
        f"(default: {DEFAULT_MAX_SLEEP_INTERVAL_SECONDS}). Only used if "
        "--sleep-interval is also set (non-negative).",
    )
    parser.add_argument(
        "--player-clients",
        default=",".join(DEFAULT_PLAYER_CLIENTS),
        help="Comma-separated YouTube player client(s) yt-dlp should impersonate "
        f"(default: {','.join(DEFAULT_PLAYER_CLIENTS)!r}), passed via extractor_args "
        "-> youtube -> player_client. Mobile/TV clients historically don't require "
        "solving YouTube's JS challenge and are less likely to be blocked from "
        "datacenter IPs (e.g. EC2). Pass an empty string to fall back to yt-dlp's "
        "own default client selection.",
    )
    args = parser.parse_args(argv)

    query = args.query or input("Wpisz słowo/frazę do wyszukania na YouTube: ").strip()
    if not query:
        raise SystemExit("Nie podano frazy wyszukiwania.")

    max_duration_seconds = args.max_duration if args.max_duration and args.max_duration > 0 else None
    remote_components = (
        [c.strip() for c in args.remote_components.split(",") if c.strip()]
        if args.remote_components
        else None
    )
    player_clients = (
        [c.strip() for c in args.player_clients.split(",") if c.strip()]
        if args.player_clients
        else None
    )
    sleep_interval = args.sleep_interval if args.sleep_interval is not None and args.sleep_interval >= 0 else None
    max_sleep_interval = (
        args.max_sleep_interval
        if sleep_interval is not None and args.max_sleep_interval is not None and args.max_sleep_interval >= 0
        else None
    )

    if args.count == 1:
        path = search_and_download_audio(
            query,
            args.output_dir,
            args.audio_format,
            max_duration_seconds=max_duration_seconds,
            cookies_file=args.cookies_file,
            js_runtime=args.js_runtime,
            remote_components=remote_components,
            sleep_interval=sleep_interval,
            max_sleep_interval=max_sleep_interval,
            player_clients=player_clients,
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
            js_runtime=args.js_runtime,
            remote_components=remote_components,
            sleep_interval=sleep_interval,
            max_sleep_interval=max_sleep_interval,
            player_clients=player_clients,
        )
        print(f"Pobrano {len(paths)}/{args.count} plik(ów) dla zapytania '{query}'.")


if __name__ == "__main__":
    main()
