#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./slurm_output/_slurm_output_htmlreport_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_htmlreport_%j.txt
#SBATCH --qos=high

# No --gpus-per-task: create_html_report.py only reads the .nii.gz predictions
# Pullback_prediction.py already wrote and encodes JPEGs -- CPU-only, no torch/model
# involved, so it doesn't need a GPU allocation.

cd /data/diag/rubenvdw/Conditional_INR/Metrics

# pandas/openpyxl -> reading the split .xlsx; opencv-python-headless -> JPEG encoding
pip install -U pandas openpyxl opencv-python-headless

echo "Running on node: $(hostname)"

MODEL_DIR="/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense_foregroundnorm"
OVERLAP=0.5              # must match the --overlap used when Pullback_prediction.py made the predictions
USE_POSTPROCESSED=true   # true -> prefer *_postprocessed.nii.gz predictions when present

ARGS=(
    --model_dir "$MODEL_DIR"
    --env       cluster
    --overlap   "$OVERLAP"
)
if [ "$USE_POSTPROCESSED" = true ]; then
    ARGS+=(--use_postprocessed)
fi

python3 -u create_html_report.py "${ARGS[@]}"