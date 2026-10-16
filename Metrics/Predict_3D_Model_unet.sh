#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./slurm_output/_slurm_output_metrics_unet_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_metrics_unet_%j.txt
#SBATCH --qos=low

cd /data/diag/rubenvdw/Conditional_INR/Metrics

# v1.6 already bakes in pandas/openpyxl; scikit-image/opencv-python-headless are
# Metrics-specific, still need installing (see Predict_3D_Model.sh's own comment on this).
pip install -U scikit-image opencv-python-headless

echo "Running on GPU node: $(hostname)"

MODEL_DIR="/data/diag/rubenvdw/Conditional_INR/saved_models_unet2p5D_conditional/residualunet2p5d_encoder64_depth5_zarrstore_zjitter_perframe_skipconnections"
CHECKPOINT="best"     # best | latest
POSTPROCESS=true      # true -> apply small-region cleanup before scoring
HTML_REPORT=true      # true -> also render the OCT|GT|prediction HTML QC report
SKIP_EXISTING=false   # true -> reuse a pullback's prediction .nii.gz if already present instead of
                       # recomputing every prediction from scratch. false -> always recompute.

# ResidualUNet2p5D always predicts at exactly its own encoder input resolution (resize_to=512
# for every config in this sweep) -- there is no --overlap or --eval_resolution here, see
# residualunet2p5d_inference.py's docstring and Pullback_prediction_unet.py for why. Since 512
# != native (704), the postprocessing thresholds need the res512-rescaled config, not the
# native-default one Predict_3D_Model.sh uses.
POSTPROCESS_CONFIG="/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR/postprocessing_classes_conditionalinr_res512.txt"

ARGS=(
    --model_dir  "$MODEL_DIR"
    --checkpoint "$CHECKPOINT"
    --env        cluster
    --device     cuda
)
if [ "$POSTPROCESS" = true ]; then
    ARGS+=(--postprocess --postprocess_config "$POSTPROCESS_CONFIG")
fi
if [ "$HTML_REPORT" = true ]; then
    ARGS+=(--html_report)
fi
if [ "$SKIP_EXISTING" = true ]; then
    ARGS+=(--skip_existing_predictions)
fi

python3 -u Pullback_prediction_unet.py "${ARGS[@]}"
