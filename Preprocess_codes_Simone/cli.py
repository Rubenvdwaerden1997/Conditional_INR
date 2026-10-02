"""CLI: oct-fm {build-store, bench, train, eval} <experiment.yaml> [--set a.b=c ...].

Paths outside the repository are never written into a config file: every command that needs one
takes it as a required command-line flag (`--store`, and for build-store `--dst --excel
--label-root --image-root`), which is applied on top of the YAML.
"""
from __future__ import annotations

import argparse

from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="oct-fm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", help="experiment YAML (experiments/*.yaml)")
        p.add_argument("--set", action="append", default=[], metavar="a.b=c",
                       help="override one config key (repeatable)")
        if name != "build-store":
            p.add_argument("--store", required=True, help="the frame store built by build-store")
        return p

    s = command("build-store", "raw images + labels + split sheet -> frame store (run once)")
    s.add_argument("--dst", required=True, help="where the frame store is written")
    s.add_argument("--excel", required=True, help="the split sheet (.xlsx)")
    s.add_argument("--label-root", required=True, help="directory holding the label volumes")
    s.add_argument("--image-root", required=True, action="append", metavar="DIR",
                   help="directory holding the raw pullbacks (repeatable)")
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--limit", type=int, default=0, help="only the first N pullbacks")
    s.add_argument("--only", default="", help="only pullback ids containing this substring")
    s.add_argument("--dry-run", action="store_true", help="plan and print, write nothing")
    s.add_argument("--finalize-only", action="store_true",
                   help="only rebuild the tables / dataset_info.json from what is already written")

    b = command("bench", "dataloader throughput of the experiment's train loader")
    b.add_argument("--seconds", type=float, default=60.0)

    t = command("train", "run the experiment (task: pretrain | seg); resumes automatically")
    t.add_argument("--resume", default="auto", help="'auto', 'none', or a checkpoint path")

    e = command("eval", "evaluate a checkpoint: seg -> per-class Dice/IoU; pretrain -> health metrics")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="test", help="seg only")
    e.add_argument("--z-consistency", action="store_true",
                   help="seg only: also predict every pullback of the split and score z-consistency")
    e.add_argument("--pullback", default="", help="seg only: also predict this whole pullback")
    e.add_argument("--out", default="", help="seg only: write that predicted volume here (.nii.gz)")

    args = ap.parse_args()
    cfg = load_config(args.config, args.set, _path_flags(args))
    resume = None if getattr(args, "resume", "auto") == "none" else getattr(args, "resume", "auto")

    if args.cmd == "build-store":
        from .store.build import main as build_store_main
        build_store_main(cfg, jobs=args.jobs, limit=args.limit, only=args.only,
                         dry_run=args.dry_run, finalize_only=args.finalize_only)
    elif args.cmd == "bench":
        from .data.frame_loader import bench_main
        bench_main(cfg, seconds=args.seconds)
    elif args.cmd == "train" and cfg.task == "pretrain":
        from .train.pretrain_loop import main as pretrain_main
        pretrain_main(cfg, resume=resume)
    elif args.cmd == "train":
        from .train.seg_loop import main as train_seg_main
        train_seg_main(cfg, resume=resume)
    elif args.cmd == "eval" and cfg.task == "pretrain":
        from .evals.pretrain_metrics import main as eval_pretrain_main
        eval_pretrain_main(cfg, ckpt=args.ckpt)
    elif args.cmd == "eval":
        from .evals.seg_metrics import main as eval_seg_main
        eval_seg_main(cfg, ckpt=args.ckpt, split=args.split, pullback=args.pullback, out=args.out,
                      z_consistency=args.z_consistency)


def _path_flags(args: argparse.Namespace) -> dict:
    """The required path flags as config keys, applied after the YAML and any `--set`."""
    if args.cmd == "build-store":
        return {"store.dst": args.dst, "store.excel": args.excel,
                "store.label_root": args.label_root,
                "store.image_roots": tuple(args.image_root)}
    return {"frames.store": args.store}


if __name__ == "__main__":
    main()
