"""Tests for the dataset downloader.

The download itself is monkeypatched, so these tests never touch the network.
"""

import io
import zipfile
from pathlib import Path

import pytest

from spam_detector import data
from spam_detector.data import ensure_dataset, extract_dataset, main

PAYLOAD = b"ham\tcall me tomorrow\nspam\tfree prize claim now\n"


def make_archive(members: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive from ``{name: bytes}``."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    return buffer.getvalue()


@pytest.fixture
def archive_bytes():
    return make_archive({"SMSSpamCollection": PAYLOAD})


def test_extract_dataset_returns_the_member_payload(archive_bytes):
    assert extract_dataset(archive_bytes) == PAYLOAD


def test_extract_dataset_finds_a_nested_member():
    nested = make_archive({"sms+spam+collection/SMSSpamCollection": PAYLOAD})

    assert extract_dataset(nested) == PAYLOAD


def test_extract_dataset_ignores_directory_entries():
    directory_only = make_archive({"SMSSpamCollection/": b""})

    with pytest.raises(ValueError, match="one expected dataset file"):
        extract_dataset(directory_only)


def test_extract_dataset_rejects_a_missing_member():
    with pytest.raises(ValueError, match="one expected dataset file"):
        extract_dataset(make_archive({"something_else.txt": PAYLOAD}))


def test_extract_dataset_rejects_ambiguous_archives():
    ambiguous = make_archive(
        {
            "a/SMSSpamCollection": PAYLOAD,
            "b/SMSSpamCollection": PAYLOAD,
        }
    )

    with pytest.raises(ValueError, match="one expected dataset file"):
        extract_dataset(ambiguous)


def test_extract_dataset_rejects_an_empty_payload():
    with pytest.raises(ValueError, match="is empty"):
        extract_dataset(make_archive({"SMSSpamCollection": b"  \n"}))


def test_extract_dataset_rejects_an_oversized_payload(archive_bytes):
    with pytest.raises(ValueError, match="size limit"):
        extract_dataset(archive_bytes, max_bytes=8)


def test_extract_dataset_rejects_a_corrupt_archive():
    with pytest.raises(ValueError, match="not a valid zip"):
        extract_dataset(b"definitely not a zip file")


def test_ensure_dataset_writes_the_downloaded_file(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "SMSSpamCollection"
    monkeypatch.setattr(
        data,
        "fetch_archive",
        lambda *a, **k: make_archive({"SMSSpamCollection": PAYLOAD}),
    )

    result = ensure_dataset(destination)

    assert result == destination
    assert destination.read_bytes() == PAYLOAD


def test_ensure_dataset_skips_an_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "SMSSpamCollection"
    destination.write_bytes(b"already here\n")

    def fail(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("fetch_archive should not be called")

    monkeypatch.setattr(data, "fetch_archive", fail)

    assert ensure_dataset(destination) == destination
    assert destination.read_bytes() == b"already here\n"


def test_ensure_dataset_force_refreshes_the_file(tmp_path, monkeypatch):
    destination = tmp_path / "SMSSpamCollection"
    destination.write_bytes(b"stale\n")
    monkeypatch.setattr(
        data,
        "fetch_archive",
        lambda *a, **k: make_archive({"SMSSpamCollection": PAYLOAD}),
    )

    ensure_dataset(destination, force=True)

    assert destination.read_bytes() == PAYLOAD


def test_ensure_dataset_surfaces_download_failures(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(data, "fetch_archive", boom)

    with pytest.raises(OSError, match="network down"):
        ensure_dataset(tmp_path / "SMSSpamCollection")


def test_main_writes_the_dataset(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "SMSSpamCollection"
    monkeypatch.setattr(
        data,
        "fetch_archive",
        lambda *a, **k: make_archive({"SMSSpamCollection": PAYLOAD}),
    )

    main(["--output", str(destination)])

    assert destination.read_bytes() == PAYLOAD
    assert "Dataset saved to" in capsys.readouterr().out


def test_main_exits_on_failure(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("bad archive")

    monkeypatch.setattr(data, "fetch_archive", boom)

    with pytest.raises(SystemExit, match="Dataset download failed"):
        main(["--output", str(tmp_path / "SMSSpamCollection")])


def test_parse_args_defaults_to_the_configured_path():
    assert data.parse_args([]).output is None
    assert data.parse_args([]).force is False
    assert data.parse_args(["--force"]).force is True


def test_default_destination_is_inside_the_data_directory():
    assert data.dataset_path().name == "SMSSpamCollection"
    assert isinstance(data.dataset_path(), Path)
