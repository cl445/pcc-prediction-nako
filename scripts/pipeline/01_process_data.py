"""Process raw NAKO CSV files into Parquet format.

Reads raw CSV files from the NAKO data directory (configured in config.toml)
and writes cleaned Parquet files to data/processed/.

Usage:
    uv run python scripts/pipeline/01_process_data.py
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from pcc_analysis.data_processing import process_all

console = Console()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    console.rule("[bold]NAKO Data Processing")

    try:
        outputs = process_all()
    except FileNotFoundError as e:
        console.print(f"[red]{e}")
        sys.exit(1)

    console.rule("[bold green]Processing Complete")
    for name, path in outputs.items():
        console.print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
