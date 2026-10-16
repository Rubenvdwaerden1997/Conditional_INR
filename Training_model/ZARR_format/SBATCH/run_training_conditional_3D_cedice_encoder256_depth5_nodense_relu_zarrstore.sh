#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./SLURM/Slurm_output_3D/_slurm_output_conditional_inr_3d_encoder256_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D/_slurm_error_conditional_inr_3d_encoder256_%j.txt
#SBATCH --qos=vram

echo "Running on node:" $(hostname)

# No pip3 install needed -- v1.6 has zarr/numpy/pandas/pyarrow/openpyxl baked in and pinned
# (see Dockerfile_initiation/). qos=vram -- largest model in this 3D width sweep, true 3D convs
# cost more per channel than the 2.5D encoder's flat 2D convs at the same feature_size.

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/main.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/ZARR_format/Config/config_3D_conditional_cedice_encoder256_depth5_nodense_relu_zarrstore.yaml
