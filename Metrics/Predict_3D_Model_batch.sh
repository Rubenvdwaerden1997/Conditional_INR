#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=144:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./slurm_output/_slurm_output_metrics_batch_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_metrics_batch_%j.txt
#SBATCH --qos=low

# Runs Pullback_prediction.py back-to-back (one job) for every Conditional INR model from the
# encoder-width sweep that doesn't have test_metrics yet -- copy of Predict_3D_Model.sh, just
# looped over MODEL_DIRS instead of one MODEL_DIR. --time is 3x the original 48h (one run per
# model, sequential, same container/model stays loaded in the shell so no re-pull per model).
# Skips a model if its test_metrics folder already exists (guards against resubmitting this
# whole batch by accident re-running everything) -- NOT the same thing as --skip_existing_predictions
# (which skips per-PULLBACK inside one model's run); this is a per-MODEL skip, outside the
# python scripts entirely.

cd /data/diag/rubenvdw/Conditional_INR/Metrics

pip install -U scikit-image opencv-python-headless

echo "Running on GPU node: $(hostname)"

MODEL_DIRS=(
    "/data/diag/rubenvdw/Conditional_INR/saved_models_2p5D_conditional/conditional_3D_relu_cedice_trilinear_encoder16_depth5_nodense_2p5d_foregroundnorm_perpullbacknorm_zjitter_perframe"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_2p5D_conditional/conditional_3D_relu_cedice_trilinear_encoder256_depth5_nodense_2p5d_foregroundnorm_perpullbacknorm_zjitter_perframe"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense_zarrstore_perpullbacknorm_zjitter_perframe"
)

CHECKPOINT="best"     # best | latest
OVERLAP=0.5           # sliding z-patch overlap fraction [0.0-1.0]
POSTPROCESS=true      # true -> apply small-region cleanup before scoring
HTML_REPORT=true      # true -> also render the OCT|GT|prediction HTML QC report
EVAL_RESOLUTION=native  # same for all three -- native-query decoupling works identically
                        # regardless of encoder_type (3d vs 2.5d), see decode_3d in model.py

if [ "$EVAL_RESOLUTION" = "encoder" ]; then
    POSTPROCESS_CONFIG="/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR/postprocessing_classes_conditionalinr_res512.txt"
else
    POSTPROCESS_CONFIG=""
fi

overlap_suffix=$(echo "$OVERLAP" | tr -d '.')
eval_suffix=""
[ "$EVAL_RESOLUTION" = "encoder" ] && eval_suffix="_encoderres"

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    TEST_METRICS_DIR="$MODEL_DIR/test_metrics_${overlap_suffix}${eval_suffix}"
    if [ -d "$TEST_METRICS_DIR" ]; then
        echo ""
        echo "=== SKIP (test_metrics already exists): $(basename "$MODEL_DIR") ==="
        continue
    fi

    echo ""
    echo "=== Running: $(basename "$MODEL_DIR") ==="

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
done

echo ""
echo "=== Batch complete ==="
