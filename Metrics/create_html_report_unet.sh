#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./slurm_output/_slurm_output_htmlreport_unet_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_htmlreport_unet_%j.txt
#SBATCH --qos=low

# No --gpus-per-task: create_html_report.py only reads the .nii.gz predictions
# Pullback_prediction_unet.py already wrote and encodes JPEGs -- CPU-only.
#
# create_html_report.py is model-agnostic (it never builds an INR- or U-Net-specific
# Config, just reads resize_to straight from the yaml), so it's reused as-is here --
# EXCEPT its default --pred_dir/--output_dir assume Pullback_prediction.py's INR-style
# folder naming (test_metrics_<overlap><eval_suffix>/), which doesn't apply to the
# U-Net (Pullback_prediction_unet.py has no --overlap/--eval_resolution, writes plainly
# to <model_dir>/test_metrics/). So both are passed explicitly below instead of relying
# on the script's defaults.

cd /data/diag/rubenvdw/Conditional_INR/Metrics

# v1.6 already bakes in pandas/openpyxl; opencv-python-headless is Metrics-specific,
# still needs installing (see Predict_3D_Model.sh's own comment on this).
pip install -U opencv-python-headless

echo "Running on node: $(hostname)"

MODEL_DIR="/data/diag/rubenvdw/Conditional_INR/saved_models_unet2p5D_conditional/residualunet2p5d_encoder64_depth5_zarrstore_zjitter_perframe_skipconnections"
PULLBACKS=""             # e.g. "NLD-RTG-0001" to only (re)build the report for one pullback -- empty = all.
LONG_ANGLE_STEP=15       # degrees between longitudinal rotation angles (0-165 deg).

ARGS=(
    --model_dir       "$MODEL_DIR"
    --env             cluster
    --pred_dir        "$MODEL_DIR/test_metrics/predictions"
    --output_dir      "$MODEL_DIR/test_metrics/html_report"
    --use_postprocessed
    --long_angle_step "$LONG_ANGLE_STEP"
    --eval_resolution encoder
)
# --eval_resolution encoder (not the default "native"): the U-Net always predicts at
# exactly resize_to (512), never native (704) -- this tells create_html_report.py to
# downsample the ground truth/raw image to match before rendering. Without this, the
# shape-mismatch assertion at pred vs seg_eval would fail (704 vs 512).
if [ -n "$PULLBACKS" ]; then
    ARGS+=(--pullbacks "$PULLBACKS")
fi

python3 -u create_html_report.py "${ARGS[@]}"
