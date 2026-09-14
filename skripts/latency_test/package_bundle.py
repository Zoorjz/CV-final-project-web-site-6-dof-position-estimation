#!/usr/bin/env python3
"""Package the latency testing suite into a standalone ZIP distribution for Raspberry Pi.

Creates 'pi_latency_benchmark_bundle.zip' containing all scripts, bundled weights,
configurations, and documentation ready to scp/copy directly to the Pi.

Usage:
  python package_bundle.py
  python package_bundle.py --output /path/to/my_bundle.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXCLUDE_NAMES = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "pi_latency_benchmark_bundle.zip",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}


def create_bundle(output_zip: Path) -> Path:
    print(f"Creating self-contained deployment bundle -> {output_zip}")
    count = 0
    total_bytes = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(HERE.rglob("*")):
            # Skip excluded paths and files
            rel_parts = item.relative_to(HERE).parts
            if any(part in EXCLUDE_NAMES for part in rel_parts):
                continue
            if item.suffix.lower() in EXCLUDE_EXTENSIONS:
                continue
            if item.is_file():
                arcname = str(Path("latency_test") / item.relative_to(HERE))
                zf.write(item, arcname=arcname)
                count += 1
                total_bytes += item.stat().st_size
                print(f"  + {arcname} ({item.stat().st_size / 1024:.1f} KB)")

    mb_size = total_bytes / (1024 * 1024)
    zip_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"\n[Success] Packed {count} files ({mb_size:.2f} MB uncompressed -> {zip_mb:.2f} MB compressed).")
    print(f"To deploy to Raspberry Pi:")
    print(f"  scp {output_zip.name} pi@<raspberry_pi_ip>:~/")
    print(f"  ssh pi@<raspberry_pi_ip> 'unzip {output_zip.name} && cd latency_test && pip install -r requirements.txt'")
    return output_zip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "pi_latency_benchmark_bundle.zip",
        help="Output ZIP file path (default: pi_latency_benchmark_bundle.zip)",
    )
    args = parser.parse_args()
    create_bundle(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
