"""Extract bidirectional flows from PCAP files using one fixed CICFlowMeter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PCAP_SUFFIXES = {".pcap", ".pcapng"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_pcaps(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES
    ]

    files.sort()

    if not files:
        raise FileNotFoundError(f"No PCAP/PCAPNG files found in: {directory}")

    return files


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        rows = sum(1 for _ in stream)

    # remove header
    return max(0, rows - 1)


def extract_one(
    pcap_path: Path,
    output_csv: Path,
    executable: str,
) -> dict:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_csv.with_suffix(".tmp.csv")

    command = [
        executable,
        "-f",
        str(pcap_path),
        "-c",
        str(temporary),
    ]

    print(f"Extracting: {pcap_path.name}")

    subprocess.run(
        command,
        check=True,
    )

    if not temporary.exists():
        raise RuntimeError(f"CICFlowMeter did not create: {temporary}")

    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Empty CICFlowMeter output: {temporary}")

    temporary.replace(output_csv)

    return {
        "input_pcap": str(pcap_path),
        "input_sha256": sha256_file(pcap_path),
        "output_csv": str(output_csv),
        "output_sha256": sha256_file(output_csv),
        "flow_rows": count_csv_rows(output_csv),
        "command": command,
    }


def extract_directory(
    domain: str,
    input_dir: Path,
    output_dir: Path,
    executable: str,
    extractor_version: str,
):
    # Prefer the CLI installed alongside the Python running this script.
    # Invoking .venv/bin/python does not itself add .venv/bin to PATH.
    local_cli = Path(sys.executable).parent / executable
    executable_path = (
        shutil.which(str(local_cli))
        if executable == "cicflowmeter" else None
    ) or shutil.which(executable)

    if executable_path is None:
        raise FileNotFoundError(
            f"Cannot find executable: {executable}. "
            "Install the cloned extractor with: "
            f"{sys.executable} -m pip install ./cicflowmeter "
            "(run from the KLTN project directory), "
            "or pass --executable /path/to/cicflowmeter."
        )

    pcaps = find_pcaps(input_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for index, pcap_path in enumerate(pcaps):

        relative = pcap_path.relative_to(input_dir)

        safe_name = "__".join(relative.parts)

        output_csv = output_dir / f"{index:04d}_{safe_name}.csv"

        record = extract_one(
            pcap_path=pcap_path,
            output_csv=output_csv,
            executable=executable_path,
        )

        records.append(record)

    manifest = {
        "domain": domain,
        "extractor": "cicflowmeter",
        "extractor_version": extractor_version,
        "executable": executable_path,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "pcap_files": len(records),
        "total_flows": sum(item["flow_rows"] for item in records),
        "files": records,
    }

    manifest_path = output_dir / "manifest.json"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            manifest,
            stream,
            indent=2,
        )

    print(f"\nFinished {domain}")
    print(f"PCAP files: {len(records)}")
    print(f"Flows: {manifest['total_flows']}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        required=True,
        choices=[
            "unsw",
            "cicids",
        ],
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--executable",
        default="cicflowmeter",
    )

    parser.add_argument(
        "--extractor-version",
        required=True,
        help=(
            "Pinned CICFlowMeter version or git commit. "
            "Must be identical for both datasets."
        ),
    )

    args = parser.parse_args()

    extract_directory(
        domain=args.domain,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        executable=args.executable,
        extractor_version=args.extractor_version,
    )


if __name__ == "__main__":
    main()
