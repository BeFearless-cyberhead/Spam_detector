"""Download the UCI SMS Spam Collection dataset used for training.

Usage::

    python -m spam_detector.data            # or: spam-detector-data
    python -m spam_detector.data --force    # re-download even if present

The archive is fetched once and written as ``label<TAB>message`` text to
``data/SMSSpamCollection``, which is the format :mod:`spam_detector.train`
expects. Downloading is split into pure helpers (:func:`fetch_archive`,
:func:`extract_dataset`) so the extraction rules can be unit tested without
network access.
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from spam_detector.config import (
    DATASET_FILENAME,
    DATASET_URL,
    dataset_path,
)

#: Refuse anything larger than this to avoid pathological downloads.
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

DOWNLOAD_TIMEOUT_SECONDS = 30


def fetch_archive(
    url: str = DATASET_URL,
    *,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> bytes:
    """Download a zip archive into memory.

    Args:
        url: Archive location.
        timeout: Socket timeout in seconds.
        max_bytes: Hard cap on the download size.

    Returns:
        The raw archive bytes.

    Raises:
        ValueError: If the payload exceeds ``max_bytes``.
        OSError: If the request fails.
    """
    with urlopen(url, timeout=timeout) as response:
        archive_bytes = response.read(max_bytes + 1)

    if len(archive_bytes) > max_bytes:
        raise ValueError("Download exceeds the expected size limit.")

    return archive_bytes


def extract_dataset(
    archive_bytes: bytes,
    *,
    member_name: str = DATASET_FILENAME,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> bytes:
    """Pull the single expected dataset member out of an archive.

    Args:
        archive_bytes: Raw zip payload.
        member_name: Base name of the expected member.
        max_bytes: Maximum accepted size for the extracted file.

    Returns:
        The dataset contents.

    Raises:
        ValueError: If the archive is invalid, does not contain exactly one
            matching file, is oversized, or is empty.
    """
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            matches = [
                item
                for item in archive.infolist()
                if Path(item.filename).name == member_name and not item.is_dir()
            ]

            if len(matches) != 1:
                raise ValueError(
                    "The archive does not contain one expected dataset file."
                )

            member = matches[0]

            if member.file_size > max_bytes:
                raise ValueError("Dataset exceeds the expected size limit.")

            dataset_bytes = archive.read(member)
    except BadZipFile as exc:
        raise ValueError("The downloaded archive is not a valid zip file.") from exc

    if not dataset_bytes.strip():
        raise ValueError("The downloaded dataset is empty.")

    return dataset_bytes


def ensure_dataset(
    path: Path | None = None,
    *,
    url: str = DATASET_URL,
    force: bool = False,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Path:
    """Make sure a usable dataset exists on disk.

    Args:
        path: Destination file; defaults to the configured dataset path.
        url: Archive location.
        force: Download again even when the file already exists.
        max_bytes: Size cap applied to the download and the extracted member.

    Returns:
        The path that now holds the dataset.

    Raises:
        ValueError: If the archive cannot be turned into a dataset.
        OSError: If the download or the write fails.
    """
    destination = Path(path) if path is not None else dataset_path()

    if destination.exists() and not force:
        print(f"Dataset already exists: {destination}")
        return destination

    print("Downloading the UCI SMS Spam Collection...")
    dataset_bytes = extract_dataset(
        fetch_archive(url, max_bytes=max_bytes),
        max_bytes=max_bytes,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(dataset_bytes)

    print(f"Dataset saved to: {destination}")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download the UCI SMS Spam Collection dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination file (default: data/SMSSpamCollection).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the destination already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Download and extract the dataset if it is not already present.

    Args:
        argv: Optional argument list, forwarded to :func:`parse_args`.

    Raises:
        SystemExit: If the download or extraction fails.
    """
    args = parse_args(argv)

    try:
        ensure_dataset(args.output, force=args.force)
    except (OSError, URLError, ValueError) as exc:
        sys.exit(f"Dataset download failed: {exc}")


if __name__ == "__main__":
    main()
