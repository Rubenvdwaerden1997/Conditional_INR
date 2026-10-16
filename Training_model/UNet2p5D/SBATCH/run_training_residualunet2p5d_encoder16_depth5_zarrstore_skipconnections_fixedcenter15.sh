#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./SLURM/Slurm_output_3D/_slurm_output_residualunet2p5d_encoder16_skipconnections_fixedcenter15_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D/_slurm_error_residualunet2p5d_encoder16_skipconnections_fixedcenter15_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-jolteon,dlc-tornadus,dlc-articuno,dlc-lugia

echo "Running on node:" $(hostname)

# No pip3 install needed -- v1.6 has zarr/numpy/pandas/pyarrow/openpyxl baked in and pinned
# (see Dockerfile_initiation/).

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/train_unet.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/UNet2p5D/Config/config_3D_residualunet2p5d_encoder16_depth5_zarrstore_skipconnections_fixedcenter15.yaml
