#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./SLURM/Slurm_output_3D/_slurm_output_conditional_inr_zarrstore_enc64_ms_l34_mid_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D/_slurm_error_conditional_inr_zarrstore_enc64_ms_l34_mid_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-jolteon,dlc-tornadus,dlc-articuno,dlc-lugia

echo "Running on node:" $(hostname)

# No pip3 install needed -- v1.6 has zarr/numpy/pandas/pyarrow/openpyxl baked in and pinned
# (see Dockerfile_initiation/), verified working via the 2.5D zarr INR run on this same image.

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/main.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/Config/Conditional_foregroundnorm/config_3D_conditional_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l34_mid.yaml
