"""Unit tests for downloader.youtube_downloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from downloader.youtube_downloader import download_audio, search_and_download_audio


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
