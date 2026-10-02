"""DataLoader factory for the frame store, and the `oct-fm bench` throughput benchmark."""
from __future__ import annotations

import multiprocessing
import time
from typing import Iterator

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, RandomSampler, Subset

from ..config import Config
from .frame_dataset import FrameWindowDataset
from .labels import class_presence
from .sampling import build_weighted_sampler
from .store import OCTStore


def build_frame_dataset(cfg: Config, split: str, train: bool, transform=None,
                        with_label: bool = True) -> FrameWindowDataset:
    """`FrameWindowDataset` over `cfg.frames.store`, with `cfg.store.center` as the polar disc centre."""
    return FrameWindowDataset(OCTStore(cfg.frames.store), cfg.frames, split, train, transform=transform,
                              center=cfg.store.center, with_label=with_label, seed=cfg.train.seed)


def build_frame_loader(cfg: Config, split: str = "train", train: bool | None = None, transform=None,
                       with_label: bool = True, batch_size: int | None = None, every: int = 1) -> DataLoader:
    """Dataset -> sampler -> DataLoader for one split.

    `train=None` means `split == "train"`. A train loader draws with a `WeightedRandomSampler` when
    `cfg.frames.sampler` is not `uniform`, else it shuffles, and it drops the last short batch; an eval loader
    (`train=False`) walks the dataset in frame-table order and keeps every item, or every `every`-th one.
    `batch_size` overrides `cfg.frames.batch_size`. `cfg.train.seed` seeds shuffling and the weighted sampler.
    """
    fc = cfg.frames
    train = (split == "train") if train is None else bool(train)
    every = int(every)
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    if every > 1 and train:
        raise ValueError("every only applies to an eval loader; a train loader samples the whole dataset")
    dataset = build_frame_dataset(cfg, split, train, transform=transform, with_label=with_label)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.train.seed))
    sampler = _train_sampler(dataset, cfg, generator) if train else None
    if every > 1:
        dataset = Subset(dataset, list(range(0, len(dataset), every)))
    kwargs = {"batch_size": fc.batch_size if batch_size is None else int(batch_size),
              "sampler": sampler,
              "shuffle": train and sampler is None,
              "drop_last": train,
              "num_workers": int(fc.num_workers),
              # a pinned copy is wasted work (and torch warns) without a device to copy to
              "pin_memory": bool(fc.pin_memory) and torch.cuda.is_available(),
              "worker_init_fn": seed_worker,
              "generator": generator}
    if int(fc.num_workers) > 0:      # torch rejects these three with num_workers=0
        kwargs |= {"prefetch_factor": int(fc.prefetch_factor),
                   "persistent_workers": bool(fc.persistent_workers),
                   "multiprocessing_context": multiprocessing.get_context(fc.mp_context)}
    return DataLoader(dataset, **kwargs)


def infinite_batches(loader: DataLoader) -> Iterator[dict]:
    """Epoch after epoch, forever (the trainers count steps, not epochs)."""
    while True:
        yield from loader


def seed_worker(worker_id: int) -> None:
    """Per-worker numpy seed from `torch.initial_seed()`, plus a modest zarr thread budget.

    torch gives every worker (and every epoch) its own base seed but does not touch numpy, so a numpy-based
    transform would otherwise replay one stream in every worker. zarr's defaults assume a single process owns
    the machine; with `num_workers` processes reading at once, 2 io threads and 8 concurrent requests each is
    plenty and leaves the cores to the decompressor.
    """
    np.random.seed(int(torch.initial_seed()) % 2**32)
    zarr.config.set({"threading.max_workers": 2, "async.concurrency": 8})


def bench_main(cfg: Config, seconds: float = 60.0) -> dict:
    """Iterate the experiment's train loader for `seconds` and report throughput (`oct-fm bench`).

    Returns `samples_per_s`, `mib_per_s` (tensor bytes of every batch), `batch_wait_p50_s`,
    `batch_wait_p95_s` and `n_samples`, and prints them with the per-class sampling share of the configured
    sampler against uniform. One batch is fetched before the clock starts, so worker start-up is not counted.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    if cfg.task == "pretrain":
        from .clips import build_clip_loader
        loader = build_clip_loader(cfg, train=True)
    else:
        loader = build_frame_loader(cfg, split="train")
    dataset: FrameWindowDataset = loader.dataset
    fc = cfg.frames
    print(f"bench: {fc.store}  task={cfg.task}")
    print(f"  items={len(dataset)} window={dataset.window} stride={list(dataset.strides)} "
          f"grid={dataset.grid_shape} label={dataset.with_label} sampler={fc.sampler}")
    print(f"  batch={loader.batch_size} workers={loader.num_workers} mp_context={fc.mp_context} "
          f"prefetch={loader.prefetch_factor} pin_memory={loader.pin_memory} draws={len(loader.sampler)}")

    it = iter(loader)
    if next(it, None) is None:
        raise RuntimeError(f"the train loader yields no batch: {len(dataset)} items, "
                           f"batch_size {loader.batch_size}, drop_last {loader.drop_last}")
    n_samples, n_bytes, waits = 0, 0, []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        asked = time.perf_counter()
        batch = next(it, None)
        if batch is None:                           # epoch exhausted
            if time.perf_counter() - start >= seconds:
                break
            it = iter(loader)
            continue
        waits.append(time.perf_counter() - asked)
        n_samples += int(batch["frame"].shape[0])
        n_bytes += sum(v.numel() * v.element_size() for v in batch.values() if torch.is_tensor(v))
    elapsed = time.perf_counter() - start
    if not waits:
        raise RuntimeError(f"no batch completed within {seconds} s; increase --seconds")

    result = {"samples_per_s": n_samples / elapsed,
              "mib_per_s": n_bytes / 2**20 / elapsed,
              "batch_wait_p50_s": float(np.percentile(waits, 50)),
              "batch_wait_p95_s": float(np.percentile(waits, 95)),
              "n_samples": n_samples}
    print(f"throughput: {result['samples_per_s']:.1f} samples/s, {result['mib_per_s']:.0f} MiB/s decoded "
          f"({n_samples} samples in {len(waits)} batches over {elapsed:.1f} s)")
    print(f"batch wait: p50 {1e3 * result['batch_wait_p50_s']:.1f} ms, "
          f"p95 {1e3 * result['batch_wait_p95_s']:.1f} ms")
    if dataset.with_label:
        _print_class_shares(dataset)
    return result


def class_sampling_shares(dataset: FrameWindowDataset) -> dict[str, tuple[float, float]]:
    """{mapped class name: (uniform share, weighted share)} for the classes that occur at all: the
    probability that one drawn sample's anchor frame contains the class."""
    present = class_presence(dataset.frames, dataset.class_map).astype(np.float64)   # [N, C]
    if not len(present):
        return {}
    weights = np.ones(len(present)) if dataset.weights is None else np.asarray(dataset.weights, dtype=np.float64)
    weighted = weights @ present / weights.sum()
    uniform = present.mean(axis=0)
    return {name: (float(uniform[c]), float(weighted[c]))
            for c, name in enumerate(dataset.class_map.names) if uniform[c] > 0}


def _train_sampler(dataset: FrameWindowDataset, cfg: Config, generator: torch.Generator):
    """The train sampler, or None when the DataLoader's own shuffle over one full epoch is what is wanted."""
    num_samples = int(cfg.frames.samples_per_epoch) or len(dataset)
    if dataset.weights is not None:
        return build_weighted_sampler(dataset.weights, num_samples, cfg.train.seed)
    if num_samples != len(dataset):
        return RandomSampler(dataset, num_samples=num_samples, generator=generator)
    return None


def _print_class_shares(dataset: FrameWindowDataset) -> None:
    scheme = dataset.cfg.sampler if dataset.weights is not None else "uniform"
    print(f"expected sampling share per class (uniform vs {scheme}):")
    print(f"  {'class':<24}{'uniform':>9}{'weighted':>11}{'ratio':>9}")
    for name, (uniform, weighted) in class_sampling_shares(dataset).items():
        print(f"  {name:<24}{100 * uniform:>8.1f}%{100 * weighted:>10.1f}%{weighted / uniform:>8.2f}x")
