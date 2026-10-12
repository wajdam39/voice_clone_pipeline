"""Unit tests for downloader.youtube_downloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from downloader.youtube_downloader import (
    download_audio,
    search_and_download_audio,
    search_and_download_many,
)


def test_download_audio_returns_expected_path(tmp_path):
    output_dir = tmp_path / "raw"
    expected_final_path = output_dir / "My Video.wav"

    fake_info = {"title": "My Video", "ext": "webm"}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.extract_info.return_value = fake_info
    mock_ydl_instance.prepare_filename.return_value = str(output_dir / "My Video.webm")
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False

    def fake_download_side_effect(*args, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_final_path.touch()

    mock_ydl_instance.extract_info.side_effect = lambda *a, **k: (
        fake_download_side_effect() or fake_info
    )

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        result_path = download_audio(
            url="https://www.youtube.com/watch?v=dummy",
            output_dir=str(output_dir),
            audio_format="wav",
        )

    assert Path(result_path) == expected_final_path
    assert expected_final_path.exists()


def test_download_audio_raises_if_file_missing(tmp_path):
    output_dir = tmp_path / "raw"

    fake_info = {"title": "Missing Video", "ext": "webm"}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.extract_info.return_value = fake_info
    mock_ydl_instance.prepare_filename.return_value = str(output_dir / "Missing Video.webm")
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        with pytest.raises(RuntimeError):
            download_audio(
                url="https://www.youtube.com/watch?v=dummy",
                output_dir=str(output_dir),
                audio_format="wav",
            )


def test_search_and_download_audio_returns_expected_path(tmp_path):
    output_dir = tmp_path / "raw"
    expected_final_path = output_dir / "Best Match.wav"

    entry = {"title": "Best Match", "ext": "webm"}
    fake_search_result = {"entries": [entry]}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False
    mock_ydl_instance.prepare_filename.return_value = str(output_dir / "Best Match.webm")

    def fake_extract_info(search_term, download=True):
        assert search_term.startswith("ytsearch")
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_final_path.touch()
        return fake_search_result

    mock_ydl_instance.extract_info.side_effect = fake_extract_info

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        result_path = search_and_download_audio(
            query="Artist - Song Title",
            output_dir=str(output_dir),
            audio_format="wav",
        )

    assert Path(result_path) == expected_final_path
    assert expected_final_path.exists()


def test_search_and_download_audio_raises_on_empty_query(tmp_path):
    with pytest.raises(ValueError):
        search_and_download_audio(query="   ", output_dir=str(tmp_path))


def test_search_and_download_audio_raises_when_no_results(tmp_path):
    output_dir = tmp_path / "raw"

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False
    mock_ydl_instance.extract_info.return_value = {"entries": []}

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        with pytest.raises(RuntimeError):
            search_and_download_audio(
                query="an obscure query with no results",
                output_dir=str(output_dir),
            )


def test_search_and_download_many_downloads_all_results(tmp_path):
    output_dir = tmp_path / "raw"

    entries = [{"title": f"Track {i}", "ext": "webm", "webpage_url": f"https://yt/{i}"} for i in range(5)]
    search_result = {"entries": entries}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False

    created_files = []

    def fake_extract_info(target, download=False):
        if target.startswith("ytsearch"):
            return search_result
        # target is a per-video URL; find the matching entry by url
        entry = next(e for e in entries if e["webpage_url"] == target)
        final_path = output_dir / f"{entry['title']}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path.touch()
        created_files.append(final_path)
        return entry

    def fake_prepare_filename(info):
        return str(output_dir / f"{info['title']}.webm")

    mock_ydl_instance.extract_info.side_effect = fake_extract_info
    mock_ydl_instance.prepare_filename.side_effect = fake_prepare_filename

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        result_paths = search_and_download_many(
            query="some topic",
            output_dir=str(output_dir),
            count=5,
            audio_format="wav",
        )

    assert len(result_paths) == 5
    assert len(created_files) == 5


def test_search_and_download_many_skips_failed_entries(tmp_path):
    output_dir = tmp_path / "raw"

    entries = [
        {"title": "Good Track", "ext": "webm", "webpage_url": "https://yt/good"},
        {"title": "Bad Track", "ext": "webm", "webpage_url": "https://yt/bad"},
    ]
    search_result = {"entries": entries}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False

    def fake_extract_info(target, download=False):
        if target.startswith("ytsearch"):
            return search_result
        if target == "https://yt/bad":
            raise RuntimeError("simulated download failure")
        entry = next(e for e in entries if e["webpage_url"] == target)
        final_path = output_dir / f"{entry['title']}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path.touch()
        return entry

    def fake_prepare_filename(info):
        return str(output_dir / f"{info['title']}.webm")

    mock_ydl_instance.extract_info.side_effect = fake_extract_info
    mock_ydl_instance.prepare_filename.side_effect = fake_prepare_filename

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        result_paths = search_and_download_many(
            query="some topic",
            output_dir=str(output_dir),
            count=2,
            audio_format="wav",
            skip_errors=True,
        )

    assert len(result_paths) == 1
    assert "Good Track" in result_paths[0]


def test_search_and_download_many_raises_on_invalid_count(tmp_path):
    with pytest.raises(ValueError):
        search_and_download_many(query="x", output_dir=str(tmp_path), count=0)


def test_search_and_download_many_filters_out_non_video_entries(tmp_path):
    """A ytsearch result page can include non-video entries (e.g. a channel
    page, ie_key='YoutubeTab') mixed in with real videos. These must be
    filtered out before downloading, since passing a channel/playlist URL to
    extract_info(download=True) would recursively pull many/all of its
    videos instead of the single result it appears to be.
    """
    output_dir = tmp_path / "raw"

    entries = [
        {"title": "Real Video", "ext": "webm", "webpage_url": "https://yt/real", "ie_key": "Youtube"},
        {
            "title": "Some Channel",
            "ext": None,
            "webpage_url": "https://www.youtube.com/channel/UCxxxx",
            "ie_key": "YoutubeTab",
        },
    ]
    search_result = {"entries": entries}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False

    extracted_urls = []

    def fake_extract_info(target, download=False):
        if target.startswith("ytsearch"):
            return search_result
        extracted_urls.append(target)
        entry = next(e for e in entries if e["webpage_url"] == target)
        final_path = output_dir / f"{entry['title']}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path.touch()
        return entry

    def fake_prepare_filename(info):
        return str(output_dir / f"{info['title']}.webm")

    mock_ydl_instance.extract_info.side_effect = fake_extract_info
    mock_ydl_instance.prepare_filename.side_effect = fake_prepare_filename

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        result_paths = search_and_download_many(
            query="some topic",
            output_dir=str(output_dir),
            count=2,
            audio_format="wav",
        )

    # Only the real video should ever have been passed to extract_info for download.
    assert extracted_urls == ["https://yt/real"]
    assert len(result_paths) == 1
    assert "Real Video" in result_paths[0]


def test_search_and_download_many_stops_exactly_at_count(tmp_path):
    """Even if more candidate video entries are available than requested,
    the function must stop downloading as soon as ``count`` successes are
    reached, instead of continuing through the rest of the result list.
    """
    output_dir = tmp_path / "raw"

    entries = [
        {"title": f"Track {i}", "ext": "webm", "webpage_url": f"https://yt/{i}", "ie_key": "Youtube"}
        for i in range(10)
    ]
    search_result = {"entries": entries}

    mock_ydl_instance = MagicMock()
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance
    mock_ydl_instance.__exit__.return_value = False

    extracted_urls = []

    def fake_extract_info(target, download=False):
        if target.startswith("ytsearch"):
            return search_result
        extracted_urls.append(target)
        entry = next(e for e in entries if e["webpage_url"] == target)
        final_path = output_dir / f"{entry['title']}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path.touch()
        return entry

    def fake_prepare_filename(info):
        return str(output_dir / f"{info['title']}.webm")

    mock_ydl_instance.extract_info.side_effect = fake_extract_info
    mock_ydl_instance.prepare_filename.side_effect = fake_prepare_filename

    fake_yt_dlp_module = MagicMock()
    fake_yt_dlp_module.YoutubeDL.return_value = mock_ydl_instance

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp_module}):
        result_paths = search_and_download_many(
            query="some topic",
            output_dir=str(output_dir),
            count=2,
            audio_format="wav",
        )

    # Exactly 2 downloads attempted/succeeded, even though 10 candidates existed.
    assert len(extracted_urls) == 2
    assert len(result_paths) == 2
