"""Verify a reviewed binary inventory against local release wheels (no network)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lato2_modly.binary_wheels import INDEX, load_inventory, verify_wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing the release wheels')
    parser.add_argument('--inventory', type=Path, default=INDEX, help='Reviewed local inventory; never a remote URL')
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error('directory must be a directory')
    inventory = load_inventory(args.inventory.resolve(strict=True))
    if not inventory:
        parser.error('inventory must not be empty')
    for spec in inventory:
        verify_wheel(directory / spec.filename, spec)
        print(f'Verified {spec.key}: {spec.sha256}')
    print(f'All {len(inventory)} wheels match the reviewed inventory.')


if __name__ == '__main__':
    main()
