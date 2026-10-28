#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH --qos=low
#SBATCH -o ./slurm_output/_slurm_output_hfer_batch_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_hfer_batch_%j.txt

# HFER for every model that already has full-pullback predictions on disk -- NO inference, no GPU,
# no continuous metrics / HTML report: it just reads the saved *_postprocessed.nii.gz volumes from
# <model>/test_metrics*/predictions and writes ONE Excel (HFER per pullback + summary per model).
# Use this instead of re-running the Predict_3D_Model*.sh scripts for models that are already scored.
# (New runs get HFER automatically inside Pullback_prediction*.py -- see HFER.py.)
#
# Covers every saved_models_* root below; models without a postprocessed prediction folder (or
# with no matching files) are skipped with a warning by HFER.py. Native-resolution (test_metrics_05)
# and encoder-resolution (test_metrics_05_encoderres) runs are both included and kept apart by name.
#
# Guiding-catheter/artifact frames are excluded via Continuous_3D_Metrics/Exclusion_Frames_testset.xlsx
# (HFER.py's default). Output: Metrics/HFER_output/hfer_<date>.xlsx

cd /data/diag/rubenvdw/Conditional_INR/Metrics
mkdir -p slurm_output

BASE="/data/diag/rubenvdw/Conditional_INR"
ROOTS=(
    saved_models_3D_conditional
    saved_models_2p5D_conditional
    saved_models_2D_conditional
    saved_models_unet2D_conditional
    saved_models_unet2p5D_conditional
    saved_models_unet3D_conditional
)
PATTERN="*_postprocessed.nii.gz"   # matches both {pid}_pred_overlap0.5_postprocessed.nii.gz (INR) and {pid}_pred_postprocessed.nii.gz (U-Net)

PRED_DIRS=()
MODEL_NAMES=()
for ROOT in "${ROOTS[@]}"; do
    for PDIR in "$BASE/$ROOT"/*/test_metrics*/predictions; do
        [ -d "$PDIR" ] || continue
        REL="${PDIR#$BASE/}"                 # saved_models_x/<model>/test_metrics*/predictions
        REL="${REL%/predictions}"
        PRED_DIRS+=("$PDIR")
        MODEL_NAMES+=("${REL#saved_models_}")
    done
done
echo "Found ${#PRED_DIRS[@]} prediction folders"

python3 -u HFER.py \
    --pred_dirs   "${PRED_DIRS[@]}" \
    --model_names "${MODEL_NAMES[@]}" \
    --pattern     "$PATTERN" \
    --output_dir  "$BASE/Metrics/HFER_output"
