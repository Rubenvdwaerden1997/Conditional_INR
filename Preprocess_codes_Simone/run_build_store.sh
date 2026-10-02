#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#diag/base-images:base-pt2.7.1"
#SBATCH --qos=high
#SBATCH -o ./logs/_slurm_output_build_store_%j.txt
#SBATCH -e ./logs/_slurm_error_build_store_%j.txt

# CPU-bound build (image decode + zarr writes): no --gpus-per-task requested on purpose, this
# step never touches a GPU. --jobs below should stay <= --cpus-per-task.
# Fresh base container each job (same pattern as oct-fm's train_seg.sbatch): install exact deps here
# rather than relying on a pre-baked image having the right zarr version.

echo "Running on node:" $(hostname)

pip3 install -q --no-cache-dir pandas numpy openpyxl pyyaml pyarrow simpleitk nibabel \
    opencv-python-headless "zarr>=3.1"

python3 -u /data/diag/rubenvdw/Conditional_INR/Preprocess_codes_Simone/run_build_store.py \
    --dst "/data/diag/rubenvdw/Dataset/Grayscale_Circmask_Unnormalized_OctFmZarrStore_17classes_30092026" \
    --excel "/data/diag/rubenvdw/Info_files_Dataset_split/Multiclass_segmentations_stratifiedsplit_28112025_newnamings15012026.xlsx" \
    --label-root "/data/diag/rubenvdw/Dataset/SEGMENTATIONS_17classes_18112025" \
    --image-root "/data/diag/rubenvdw/Dataset/DICOMS_Pectus" \
    --image-root "/data/diag/CARAlab_datasets/ORANGE_dicoms" \
    --jobs 10
