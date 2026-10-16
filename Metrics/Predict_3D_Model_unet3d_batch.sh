#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=168:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./slurm_output/_slurm_output_metrics_unet3d_batch_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_metrics_unet3d_batch_%j.txt
#SBATCH --qos=high
#SBATCH --exclude=dlc-articuno,dlc-lugia,dlc-tornadus

# Runs Pullback_prediction_unet3d.py back-to-back (one job) for every ResidualUNet3D model from
# the encoder-width sweep that doesn't have test_metrics yet -- copy of Predict_3D_Model_unet3d.sh,
# just looped over MODEL_DIRS instead of one MODEL_DIR. --time is 5x the original 48h (one run
# per model, sequential). Skips a model if its test_metrics folder already exists.

cd /data/diag/rubenvdw/Conditional_INR/Metrics

pip install -U scikit-image opencv-python-headless

echo "Running on GPU node: $(hostname)"

MODEL_DIRS=(
    "/data/diag/rubenvdw/Conditional_INR/saved_models_unet3D_conditional/residualunet3d_encoder16_depth5_zarrstore_zjitter_perframe_skipconnections"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_unet3D_conditional/residualunet3d_encoder32_depth5_zarrstore_zjitter_perframe_skipconnections"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_unet3D_conditional/residualunet3d_encoder64_depth5_zarrstore_zjitter_perframe_skipconnections"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_unet3D_conditional/residualunet3d_encoder128_depth5_zarrstore_zjitter_perframe_skipconnections"
    "/data/diag/rubenvdw/Conditional_INR/saved_models_unet3D_conditional/residualunet3d_encoder256_depth5_zarrstore_zjitter_perframe_skipconnections"
)

CHECKPOINT="best"     # best | latest
OVERLAP=0.5           # sliding-window overlap fraction between patch_z windows
POSTPROCESS=true      # true -> apply small-region cleanup before scoring
HTML_REPORT=true      # true -> also render the OCT|GT|prediction HTML QC report

# ResidualUNet3D always predicts at exactly its own encoder input resolution (resize_to=512
# for every config in this sweep) -- no --eval_resolution, see residualunet3d_inference.py.
POSTPROCESS_CONFIG="/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR/postprocessing_classes_conditionalinr_res512.txt"

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    TEST_METRICS_DIR="$MODEL_DIR/test_metrics"
    if [ -d "$TEST_METRICS_DIR" ]; then
        echo ""
        echo "=== SKIP (test_metrics already exists): $(basename "$MODEL_DIR") ==="
        continue
    fi

    echo ""
    echo "=== Running: $(basename "$MODEL_DIR") ==="

    ARGS=(
        --model_dir  "$MODEL_DIR"
        --checkpoint "$CHECKPOINT"
        --overlap    "$OVERLAP"
        --env        cluster
        --device     cuda
    )
    if [ "$POSTPROCESS" = true ]; then
        ARGS+=(--postprocess --postprocess_config "$POSTPROCESS_CONFIG")
    fi
    if [ "$HTML_REPORT" = true ]; then
        ARGS+=(--html_report)
    fi

    python3 -u Pullback_prediction_unet3d.py "${ARGS[@]}"
done

echo ""
echo "=== Batch complete ==="
