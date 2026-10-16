#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./SLURM/Slurm_output_3D/_slurm_output_residualunet3d_skipconnections_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D/_slurm_error_residualunet3d_skipconnections_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-articuno,dlc-lugia,dlc-tornadus

echo "Running on node:" $(hostname)

# No pip3 install needed -- v1.6 has zarr/numpy/pandas/pyarrow/openpyxl baked in and pinned.

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/train_unet3d.py     --config /data/diag/rubenvdw/Conditional_INR/Training_model/UNet3D/Config/config_3D_residualunet3d_encoder32_depth5_zarrstore_skipconnections.yaml
