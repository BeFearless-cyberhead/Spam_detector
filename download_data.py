"""Download the UCI SMS Spam Collection dataset used for training.

Usage::

    python download_data.py

The archive is fetched once and written to ``data/SMSSpamCollection`` in the
same ``label<TAB>message`` format that ``train.py`` expects.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "SMSSpamCollection"

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

#: Refuse anything larger than this to avoid pathological downloads.
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


def main() -> None:
    """Download and extract the dataset if it is not already present.

    Raises:
        SystemExit: If the download fails, exceeds the size limit, or the
            archive does not contain exactly one dataset member.
    """
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        return

    print("Downloading the UCI SMS Spam Collection...")

    try:
        with urlopen(DATA_URL, timeout=30) as response:
            archive_bytes = response.read(MAX_DOWNLOAD_BYTES + 1)

        if len(archive_bytes) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Download exceeds the expected size limit.")

        with ZipFile(BytesIO(archive_bytes)) as archive:
            matches = [
                item
                for item in archive.infolist()
                if Path(item.filename).name == "SMSSpamCollection" and not item.is_dir()
            ]

            if len(matches) != 1:
                raise ValueError(
                    "The archive does not contain one expected dataset file."
                )

            member = matches[0]

            if member.file_size > MAX_DOWNLOAD_BYTES:
                raise ValueError("Dataset exceeds the expected size limit.")

            dataset_bytes = archive.read(member)

            if not dataset_bytes.strip():
                raise ValueError("The downloaded dataset is empty.")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            DATA_PATH.write_bytes(dataset_bytes)
    except (OSError, URLError, BadZipFile, ValueError) as exc:
        sys.exit(f"Dataset download failed: {exc}")

    print(f"Dataset saved to: {DATA_PATH}")


if __name__ == "__main__":
    main()
