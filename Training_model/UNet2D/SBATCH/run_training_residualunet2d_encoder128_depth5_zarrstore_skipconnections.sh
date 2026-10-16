#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./SLURM/Slurm_output_3D/_slurm_output_residualunet2d_encoder128_skipconnections_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D/_slurm_error_residualunet2d_encoder128_skipconnections_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-tornadus,dlc-jolteon

echo "Running on node:" $(hostname)

# No pip3 install needed -- v1.6 has zarr/numpy/pandas/pyarrow/openpyxl baked in and pinned
# (see Dockerfile_initiation/). [2.5D run used qos=vram; this 2D run is on low -- 1 input frame, far less activation memory] -- encoder_feat=128 needs more activation memory.

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/train_unet.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/UNet2D/Config/config_3D_residualunet2d_encoder128_depth5_zarrstore_skipconnections.yaml
