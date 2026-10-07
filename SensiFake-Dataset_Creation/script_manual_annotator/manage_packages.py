"""Prepare, snapshot, and merge local SensiFake annotation packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from manual_annotator.script.annotation_packages import (
    merge_offline,
    prepare_offline,
    snapshot_offline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Split a batch into local databases")
    prepare.add_argument("--master", type=Path, required=True)
    prepare.add_argument("--batch", required=True)
    prepare.add_argument("--parts", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    snapshot = commands.add_parser("snapshot", help="Copy one local database for return")
    snapshot.add_argument("--package", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("merge", help="Import returned annotations into master")
    merge.add_argument("--master", type=Path, required=True)
    merge.add_argument("--assignment", type=Path, required=True)
    merge.add_argument("packages", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_offline(args.master, args.batch, args.parts, args.output)
        for entry in result["packages"]:
            print(f"{args.output / entry['package']}: {len(entry['hashes'])} images")
        print(f"Keep assignment file: {args.output / 'assignment.json'}")
    elif args.command == "snapshot":
        snapshot_offline(args.package, args.output)
        print(f"Share snapshot: {args.output}")
    else:
        print(f"Imported {merge_offline(args.master, args.assignment, args.packages)} annotations.")


if __name__ == "__main__":
    main()
