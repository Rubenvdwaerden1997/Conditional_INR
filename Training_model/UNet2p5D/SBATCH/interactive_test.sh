#!/bin/bash
# Interactive debugging shell for the 2.5D Residual U-Net config -- CPU-only, no GPU requested,
# so it queues fast (GPU nodes are the scarce/contended resource; this is all --sanity_check
# needs -- train_unet.py hardcodes device=cpu for it). NOT sbatch: run this directly from the
# login node (it blocks until a node is allocated, then drops you into a live shell there). No
# queue log files -- lets you re-run train_unet.py directly and iterate without resubmitting.
#
# Usage:
#   bash interactive_test.sh
# Then inside the shell, the fast CPU-only shape/import check -- no-skip config:
#   python3 train_unet.py --config UNet2p5D/Config/config_3D_residualunet2p5d_encoder64_depth5_zarrstore.yaml --sanity_check
# or the skip-connections config:
#   python3 train_unet.py --config UNet2p5D/Config/config_3D_residualunet2p5d_encoder64_depth5_zarrstore_skipconnections.yaml --sanity_check
# For the REAL run (needs a GPU), once sanity_check passes here, submit the matching
# run_training_residualunet2p5d_encoder64_depth5_zarrstore[_skipconnections].sh via sbatch instead.
# Type `exit` when done to release this allocation.

srun --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 --qos=high \
    --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6" \
    --container-mounts=/data/diag:/data/diag \
    --pty bash -c '
        cd /data/diag/rubenvdw/Conditional_INR/Training_model
        exec bash
    '
