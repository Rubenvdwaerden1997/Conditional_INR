#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./SLURM/Slurm_output_preprocess/_slurm_output_preprocess_frames_%j.txt
#SBATCH -e ./SLURM/Slurm_output_preprocess/_slurm_error_preprocess_frames_%j.txt
#SBATCH --qos=vram
#SBATCH --exclude=dlc-mewtwo,dlc-moltres,dlc-nidoking,dlc-scyther,dlc-lugia,dlc-articuno,dlc-slowpoke

echo "Running on node:" $(hostname)

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/preprocess_frames.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/Config/Conditional_foregroundnorm/config_3D_conditional_cedice_encoder64_depth5_nodense_relu_foregroundnorm.yaml \
    --mode 3d \
    --sets training validation testing \
    --n_procs 6