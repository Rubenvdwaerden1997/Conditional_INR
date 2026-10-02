#!/bin/bash
# Interactive debugging shell for the 2.5D+zarr training config -- CPU-only, no GPU requested,
# so it queues fast (GPU nodes are the scarce/contended resource; this is all --sanity_check
# needs -- main.py hardcodes device=cpu for it). NOT sbatch: run this directly from the login
# node (it blocks until a node is allocated, then drops you into a live shell there). No queue
# log files -- lets you re-run main.py directly and iterate without resubmitting each time.
#
# Usage:
#   bash interactive_test.sh
# Then inside the shell, the fast CPU-only shape/import check:
#   python3 main.py --config ZARR_format/Config/config_3D_conditional_cedice_encoder64_depth5_2p5d_foregroundnorm.yaml --sanity_check
# For the REAL run (needs a GPU), once sanity_check passes here, submit
# run_training_conditional_3D_cedice_encoder64_depth5_2p5d_foregroundnorm.sh via sbatch instead
# (this allocation has no GPU, so main.py's actual training loop would fail on cfg.device="cuda").
# Type `exit` when done to release this allocation.

srun --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 --qos=high \
    --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6" \
    --container-mounts=/data/diag:/data/diag \
    --pty bash -c '
        cd /data/diag/rubenvdw/Conditional_INR/Training_model
        exec bash
    '