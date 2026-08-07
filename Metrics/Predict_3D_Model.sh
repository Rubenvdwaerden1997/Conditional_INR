#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./slurm_output/_slurm_output_metrics_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_metrics_%j.txt
#SBATCH --qos=low
#SBATCH --exclude=dlc-mewtwo,dlc-moltres,dlc-nidoking,dlc-scyther,dlc-lugia,dlc-zapdos

cd /data/diag/rubenvdw/Conditional_INR/Metrics

# pandas/openpyxl -> reading the split .xlsx; scikit-image -> only exercised if POSTPROCESS=true below;
# opencv-python-headless -> Pullback_prediction.py now imports create_html_report.py's JPEG helpers at
# module load time regardless of HTML_REPORT below, so cv2 is an unconditional dependency here.
pip install -U pandas openpyxl scikit-image opencv-python-headless

echo "Running on GPU node: $(hostname)"

MODEL_DIR="/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense_foregroundnorm_coordjitter05_tversky"
CHECKPOINT="best"     # best | latest
OVERLAP=0.5           # sliding z-patch overlap fraction [0.0-1.0]
POSTPROCESS=true     # true -> apply small-region cleanup before scoring
HTML_REPORT=true      # true -> also render the OCT|GT|prediction HTML QC report (see create_html_report.py)

ARGS=(
    --model_dir  "$MODEL_DIR"
    --checkpoint "$CHECKPOINT"
    --env        cluster
    --overlap    "$OVERLAP"
    --device     cuda
)
if [ "$POSTPROCESS" = true ]; then
    ARGS+=(--postprocess)
fi
if [ "$HTML_REPORT" = true ]; then
    ARGS+=(--html_report)
fi

python3 -u Pullback_prediction.py "${ARGS[@]}"
