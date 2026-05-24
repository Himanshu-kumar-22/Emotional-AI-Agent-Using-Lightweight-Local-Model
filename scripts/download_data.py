"""
scripts/download_data.py
========================
Downloads the GoEmotions dataset from the official Google Research
GitHub repository and saves it to data/raw/.

GoEmotions is distributed as three TSV split files:
  - train.tsv   (~43K samples, 58% of total)
  - dev.tsv     (~5.4K samples, validation split)
  - test.tsv    (~5.4K samples, held-out evaluation)

Each row: text \t comma_separated_label_ids \t annotator_id

The label taxonomy file maps integer IDs to emotion names.

Usage:
    python3 scripts/download_data.py

Design note:
    We download directly from the source repository rather than using
    the HuggingFace datasets hub version because we want explicit control
    over the raw files for reproducibility and offline use.
"""

import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

# ── Source URLs ───────────────────────────────────────────────────────────────
BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research"
    "/master/goemotions/data"
)

FILES = {
    "train.tsv": f"{BASE_URL}/train.tsv",
    "dev.tsv": f"{BASE_URL}/dev.tsv",
    "test.tsv": f"{BASE_URL}/test.tsv",
    "emotions.txt": f"{BASE_URL}/emotions.txt",
}


def download_file(url: str, destination: Path, description: str) -> bool:
    """
    Download a file with a progress bar.

    Uses streaming download to handle large files without loading
    the entire response into memory.

    Returns True on success, False on failure.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(destination, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"  Downloading {description}",
                ncols=80,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

        print(f"  ✓ Saved to {destination.relative_to(PROJECT_ROOT)}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to download {description}: {e}")
        return False


def verify_download(file_path: Path) -> bool:
    """

    Validate downloaded files using file-specific rules.

    Why file-specific validation?

    --------------------------------

    Different dataset artifacts have different expected sizes.

    For example:

      - TSV dataset splits are several hundred KB to multiple MB

      - emotions.txt is intentionally tiny because it only stores labels

    Using one universal threshold causes false corruption detection.

    """

    if not file_path.exists():

        return False

    file_size_kb = file_path.stat().st_size / 1024

    filename = file_path.name

    expected_min_sizes = {
        "train.tsv": 1000,  # ~3.4 MB expected
        "dev.tsv": 100,  # ~400 KB expected
        "test.tsv": 100,  # ~400 KB expected
        "emotions.txt": 0.1,  # ~248 bytes expected
    }

    min_size = expected_min_sizes.get(filename, 1)

    return file_size_kb >= min_size


def count_lines(file_path: Path) -> int:
    """Count lines in a file efficiently."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    print("\n" + "=" * 55)
    print("  GoEmotions Dataset Downloader")
    print("=" * 55)
    print(f"\nDestination: {settings.raw_data_dir}\n")

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    all_success = True

    for filename, url in FILES.items():
        destination = settings.raw_data_dir / filename

        # Skip if already downloaded and valid
        if destination.exists() and verify_download(destination):
            size_kb = destination.stat().st_size / 1024
            print(f"  ↷ {filename} already exists ({size_kb:.0f} KB) — skipping")
            continue

        print(f"\n→ {filename}")
        success = download_file(url, destination, filename)

        if success and not verify_download(destination):
            print(f"  ✗ {filename} appears corrupted (too small)")
            destination.unlink(missing_ok=True)
            success = False

        if not success:
            all_success = False

    # Summary statistics
    print("\n" + "-" * 55)
    print("Download Summary:")

    for filename in ["train.tsv", "dev.tsv", "test.tsv"]:
        filepath = settings.raw_data_dir / filename
        if filepath.exists():
            n_lines = count_lines(filepath)
            size_kb = filepath.stat().st_size / 1024
            # TSV files have no header, so line count = sample count
            print(f"  {filename}: {n_lines:,} samples ({size_kb:.0f} KB)")

    # Show the emotion labels
    emotions_file = settings.raw_data_dir / "emotions.txt"
    if emotions_file.exists():
        emotions = emotions_file.read_text().strip().split("\n")
        print(f"\n  Emotion taxonomy: {len(emotions)} labels")
        print(f"  Labels: {', '.join(emotions[:5])}... (and {len(emotions)-5} more)")

    if all_success:
        print("\n✓ All files downloaded successfully.")
        print("  Next step: run the preprocessor\n")
    else:
        print("\n✗ Some files failed validation.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
