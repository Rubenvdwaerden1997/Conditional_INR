#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./SLURM/Slurm_output_3D_unconditional/_slurm_output_hidden32_depth3_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D_unconditional/_slurm_error_hidden32_depth3_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-mewtwo,dlc-nidoking,dlc-scyther,dlc-zapdos,dlc-moltres

echo "Running on node:" $(hostname)

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/main.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/Config/Unconditional/config_unconditional_hidden32_depth3.yaml
