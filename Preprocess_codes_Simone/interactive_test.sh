#!/bin/bash
# Interactive test shell for the build-store code, in the same base container oct-fm's
# train_seg.sbatch uses. NOT sbatch: run this directly from the login node (it blocks until
# a node is allocated, then drops you into a live shell there). No queue, no log files.
#
# Usage:
#   bash interactive_test.sh
# Then inside the shell, e.g.:
#   python3 run_build_store.py --dst "/data/diag/rubenvdw/Dataset/Grayscale_Circmask_Unnormalized_OctFmZarrStore_17classes_30092026" \
#       --excel "/data/diag/rubenvdw/Info_files_Dataset_split/Multiclass_segmentations_stratifiedsplit_28112025_newnamings15012026.xlsx" \
#       --label-root "/data/diag/rubenvdw/Dataset/SEGMENTATIONS_17classes_18112025" \
#       --image-root "/data/diag/rubenvdw/Dataset/DICOMS_Pectus" \
#       --image-root "/data/diag/CARAlab_datasets/ORANGE_dicoms" \
#       --dry-run
# Type `exit` when done to release the allocation.

srun --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 \
    --container-image="dockerdex.umcn.nl:5005#diag/base-images:base-pt2.7.1" \
    --container-mounts=/data/diag:/data/diag \
    --pty bash -c '
        pip3 install -q --no-cache-dir pandas numpy openpyxl pyyaml pyarrow simpleitk nibabel opencv-python-headless "zarr>=3.1"
        cd /data/diag/rubenvdw/Conditional_INR/Preprocess_codes_Simone
        exec bash
    '
