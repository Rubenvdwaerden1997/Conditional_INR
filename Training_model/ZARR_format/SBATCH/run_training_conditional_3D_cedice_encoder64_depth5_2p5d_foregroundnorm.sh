#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=60G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./SLURM/Slurm_output_3D/_slurm_output_conditional_inr_2p5d_%j.txt
#SBATCH -e ./SLURM/Slurm_output_3D/_slurm_error_conditional_inr_2p5d_%j.txt
#SBATCH --qos=high

echo "Running on node:" $(hostname)

# No pip3 install needed -- v1.6 has zarr/numpy/pandas/pyarrow/openpyxl baked in and pinned,
# plus an upgraded torch/torchvision build (see Dockerfile_initiation/) that fixed the
# "no kernel image is available" crash on dlc-jolteon's Blackwell/sm_120 GPU under v1.5 --
# verified on both dlc-jolteon and an older GPU node, so dlc-jolteon no longer needs excluding.

python3 -u /data/diag/rubenvdw/Conditional_INR/Training_model/main.py \
    --config /data/diag/rubenvdw/Conditional_INR/Training_model/ZARR_format/Config/config_3D_conditional_cedice_encoder64_depth5_2p5d_foregroundnorm.yaml
