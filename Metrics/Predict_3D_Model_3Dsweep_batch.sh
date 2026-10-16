#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=144:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./slurm_output/_slurm_output_metrics_3dsweep_batch_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_metrics_3dsweep_batch_%j.txt
#SBATCH --qos=low

# Runs Pullback_prediction.py back-to-back (one job) for the 3D Conditional INR encoder-width
# sweep tiers that don't have test_metrics yet -- same pattern as Predict_3D_Model_batch.sh.
# encoder64 is deliberately NOT in this list -- it already has test_metrics (done in the
# earlier batch run). encoder128 is ALSO deliberately excluded -- it was still training
# (epoch 262/300) as of 2026-10-08; add it here once it finishes rather than running metrics
# on an interim, non-final checkpoint.

cd /data/diag/rubenvdw/Conditional_INR/Metrics

pip install -U scikit-image opencv-python-headless

echo "Running on GPU node: $(hostname)"

MODEL_DIRS=(
    "/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder16_depth5_nodense_zarrstore_perpullbacknorm_zjitter_perframe"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder32_depth5_nodense_zarrstore_perpullbacknorm_zjitter_perframe"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder256_depth5_nodense_zarrstore_perpullbacknorm_zjitter_perframe"
)

CHECKPOINT="best"     # best | latest
OVERLAP=0.5           # sliding z-patch overlap fraction [0.0-1.0]
POSTPROCESS=true      # true -> apply small-region cleanup before scoring
HTML_REPORT=true      # true -> also render the OCT|GT|prediction HTML QC report
EVAL_RESOLUTION=native

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
