### Derived from the already-working, CUDA-matched training image -- not the generic diag base
### image used in other projects' Dockerfile_initiation -- specifically so torch/CUDA never needs
### to be re-resolved here, only the numpy/pandas/pyarrow/zarr/openpyxl set this project needed
### on top of it (see requirements.txt for why).
FROM dockerdex.umcn.nl:5005/rubenvdwaerden1997/train_monai:v1.4

COPY requirements.txt /tmp/

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --upgrade -r /tmp/requirements.txt

### Upgrade torch/torchvision for newer-GPU support (e.g. Blackwell / sm_120, seen failing on
### dlc-jolteon: "RuntimeError: CUDA error: no kernel image is available for execution on the
### device" -- the base image's torch only has kernels up to sm_90, so it doesn't just run slow
### on a newer GPU, it fails outright on the very first CUDA op). Kept separate from
### requirements.txt (not just appended to it) because it needs its own --index-url, and because
### every OTHER package in requirements.txt was deliberately chosen to never need to touch torch
### at all (v1.4/v1.5's whole point) -- this is the one deliberate exception, done explicitly and
### reviewably, not accidentally via a generic PyPI resolve.
### Exact versions pinned (confirmed 2026-10-02 via `docker run --rm <image> pip3 list` on the
### actual pushed v1.6) rather than left as an unpinned --upgrade, same reproducibility reasoning
### as requirements.txt -- still needs step 7's two-GPU-type test before being trusted for real.
RUN pip3 install --no-cache-dir "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip3 list

### train_monai:v1.4 (the base image) bakes in ENTRYPOINT ["/bin/bash", "/root/run.sh"] from a
### different project's own Dockerfile -- that run.sh cd's into a path and runs a script that
### only exist in that other project, not here. Reset it so `docker run <image> <cmd>` (and
### locally testing this image in general) runs <cmd> directly, like a normal image. This has no
### effect on the actual SLURM/pyxis training jobs -- those already specify the exact command to
### run in each .sh script, and pyxis/enroot execs that directly without honoring Docker
### ENTRYPOINT/CMD semantics the way `docker run` does; confirmed training already worked under
### v1.4 with this same inherited entrypoint still present. This fixes local docker run usability.
ENTRYPOINT []
