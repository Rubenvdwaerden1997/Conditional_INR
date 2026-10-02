#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./slurm_output/_slurm_output_htmlreport_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_htmlreport_%j.txt
#SBATCH --qos=low

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
EVAL_RESOLUTION=native  # must match the --eval_resolution Pullback_prediction.py made these predictions
                         # with: "native" (default) or "encoder" (scores at cfg.resize_to instead of
                         # native -- see Pullback_prediction.py --help). Output lands in its own
                         # "_encoderres"-suffixed folder + differently-named report.html, so switching
                         # this never overwrites the other mode's report.
PULLBACKS=""             # e.g. "NLD-RTG-0001" to only (re)build the report for one pullback -- much
                         # faster, handy when you just want to eyeball its longitudinal view. Empty = all.
LONG_ANGLE_STEP=15       # degrees between longitudinal rotation angles (0-165 deg); smaller = smoother
                         # rotation but proportionally more compute/output. See create_html_report.py --help.

ARGS=(
    --model_dir       "$MODEL_DIR"
    --env             cluster
    --overlap         "$OVERLAP"
    --eval_resolution "$EVAL_RESOLUTION"
    --long_angle_step "$LONG_ANGLE_STEP"
)
if [ "$USE_POSTPROCESSED" = true ]; then
    ARGS+=(--use_postprocessed)
fi
if [ -n "$PULLBACKS" ]; then
    ARGS+=(--pullbacks "$PULLBACKS")
fi

python3 -u create_html_report.py "${ARGS[@]}"