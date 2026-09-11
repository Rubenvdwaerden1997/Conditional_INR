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
#SBATCH --qos=high
#SBATCH --exclude=dlc-mewtwo,dlc-moltres,dlc-nidoking,dlc-scyther,dlc-lugia,dlc-zapdos

cd /data/diag/rubenvdw/Conditional_INR/Metrics

# pandas/openpyxl -> reading the split .xlsx; scikit-image -> only exercised if POSTPROCESS=true below;
# opencv-python-headless -> Pullback_prediction.py now imports create_html_report.py's JPEG helpers at
# module load time regardless of HTML_REPORT below, so cv2 is an unconditional dependency here.
pip install -U pandas openpyxl scikit-image opencv-python-headless

echo "Running on GPU node: $(hostname)"

MODEL_DIR="/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense_foregroundnorm_difficultyweighted_coordjitter05"
CHECKPOINT="best"     # best | latest
OVERLAP=0.5           # sliding z-patch overlap fraction [0.0-1.0]
POSTPROCESS=true     # true -> apply small-region cleanup before scoring
HTML_REPORT=true      # true -> also render the OCT|GT|prediction HTML QC report (see create_html_report.py)
EVAL_RESOLUTION=native  # "native": query the INR at native resolution, score against native ground
                        # truth. "encoder": query at the model's own encoder input resolution
                        # (cfg.resize_to) instead, scoring against nearest-neighbour-downsampled ground
                        # truth -- isolates the model's own learned quality from the native-query
                        # decoupling. Output lands in its own "_encoderres"-suffixed folder, so switching
                        # this never overwrites the other mode's predictions/metrics. Must match whatever
                        # EVAL_RESOLUTION create_html_report.sh is later run with for these predictions.

# Postprocessing thresholds are calibrated in raw pixel counts, so they must match
# EVAL_RESOLUTION's actual grid or cleanup becomes systematically more/less aggressive
# than intended (a fixed pixel_minimum covers more physical area on a coarser grid).
# The default (native, resize_to=512 model) config stays correct for native;
# encoder mode needs the values rescaled by (512/704)^2 -- see
# postprocessing_classes_conditionalinr_res512.txt in the pipeline folder.
if [ "$EVAL_RESOLUTION" = "encoder" ]; then
    POSTPROCESS_CONFIG="/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR/postprocessing_classes_conditionalinr_res512.txt"
else
    POSTPROCESS_CONFIG=""   # empty -> Pullback_prediction.py's own native-resolution default
fi

ARGS=(
    --model_dir       "$MODEL_DIR"
    --checkpoint      "$CHECKPOINT"
    --env             cluster
    --overlap         "$OVERLAP"
    --device          cuda
    --eval_resolution "$EVAL_RESOLUTION"
)
if [ "$POSTPROCESS" = true ]; then
    ARGS+=(--postprocess)
    if [ -n "$POSTPROCESS_CONFIG" ]; then
        ARGS+=(--postprocess_config "$POSTPROCESS_CONFIG")
    fi
fi
if [ "$HTML_REPORT" = true ]; then
    ARGS+=(--html_report)
fi

python3 -u Pullback_prediction.py "${ARGS[@]}"
