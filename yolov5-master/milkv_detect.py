"""Compatibility entry point; prefer the root strawberry.py CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strawberry import main

if __name__ == "__main__":
    raise SystemExit(main(["detect", *sys.argv[1:]]))
