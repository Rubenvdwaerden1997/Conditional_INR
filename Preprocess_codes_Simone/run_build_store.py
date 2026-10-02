"""Standalone entry point for the copied `oct-fm build-store` step (see store/build.py).

Builds a Cartesian zarr frame store from raw pullbacks + label volumes + the split sheet.
Trimmed copy of oct-fm's own `cli.py` build-store branch — same flags, same semantics.

Usage (dry run first, to see the plan without writing anything):
    python3 -u run_build_store.py \
        --dst /data/diag/rubenvdw/Dataset/<name> \
        --excel /data/diag/rubenvdw/Info_files_Dataset_split/Multiclass_segmentations_stratifiedsplit_28112025_newnamings15012026.xlsx \
        --label-root /data/diag/rubenvdw/Dataset/SEGMENTATIONS_17classes_18112025 \
        --image-root /data/diag/rubenvdw/Dataset/DICOMS_Pectus \
        --image-root /data/diag/CARAlab_datasets/ORANGE_dicoms \
        --dry-run

Then re-run the same command without --dry-run (add --jobs N) to actually build; re-running
the same command again resumes an interrupted build.
"""
from __future__ import annotations

import argparse
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from Preprocess_codes_Simone.config import load_config
from Preprocess_codes_Simone.store.build import main as build_store_main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="optional experiment YAML (defaults are fine)")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=c",
                    help="override one config key (repeatable)")
    ap.add_argument("--dst", required=True, help="where the frame store is written")
    ap.add_argument("--excel", required=True, help="the split sheet (.xlsx)")
    ap.add_argument("--label-root", required=True, help="directory holding the label volumes")
    ap.add_argument("--image-root", required=True, action="append", metavar="DIR",
                    help="directory holding the raw pullbacks (repeatable)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pullbacks")
    ap.add_argument("--only", default="", help="only pullback ids containing this substring")
    ap.add_argument("--dry-run", action="store_true", help="plan and print, write nothing")
    ap.add_argument("--finalize-only", action="store_true",
                    help="only rebuild the tables / dataset_info.json from what is already written")
    args = ap.parse_args()

    values = {"store.dst": args.dst, "store.excel": args.excel,
              "store.label_root": args.label_root, "store.image_roots": tuple(args.image_root)}
    cfg = load_config(args.config, args.set, values)

    build_store_main(cfg, jobs=args.jobs, limit=args.limit, only=args.only,
                     dry_run=args.dry_run, finalize_only=args.finalize_only)


if __name__ == "__main__":
    main()
