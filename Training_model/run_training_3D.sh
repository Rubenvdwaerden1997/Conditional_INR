#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./Slurm_output_3D/_slurm_output_conditional_inr_%j.txt
#SBATCH -e ./Slurm_output_3D/_slurm_error_conditional_inr_%j.txt
#SBATCH --qos=high
#SBATCH --exclude=dlc-mewtwo,dlc-nidoking,dlc-scyther,dlc-zapdos,dlc-articuno,dlc-lugia,dlc-moltres

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/main.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/Config/config_3D.yaml
