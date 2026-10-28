#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.6"
#SBATCH -o ./slurm_output/_slurm_output_htmlreport_batch_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_htmlreport_batch_%j.txt
#SBATCH --qos=low

# Builds the HTML QC report INCLUDING the rotating longitudinal (L-mode) view for every stage-1
# model below whose report is missing that view -- i.e. the first model of each
# Predict_3D_Model_stage1_*.sh job, which started before Pullback_prediction.py built the
# longitudinal view by default. Reads the existing predictions only (no GPU, no inference).
#
# A model is skipped (with a message) when:
#   - its prediction job has not finished (no report file written yet), or
#   - its report already contains a longitudinal view (nothing to do).
# So this is safe to run after the prediction jobs and safe to re-run.

cd /data/diag/rubenvdw/Conditional_INR/Metrics

pip install -U pandas openpyxl opencv-python-headless

echo "Running on node: $(hostname)"

B=/data/diag/rubenvdw/Conditional_INR/saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder
S=_depth5_nodense_zarrstore_perpullbacknorm_zjitter_perframe
MODEL_DIRS=(
    # job A -- encoder 64 core
    "${B}64${S}"
    "${B}64${S}_multiscale_l1_mid"
    # job B -- small widths
    "${B}16${S}"
    "${B}16${S}_multiscale_l1_mid"
    "${B}32${S}"
    "${B}32${S}_multiscale_l1_mid"
    # job C -- encoder 64 all-levels variants
    "${B}64${S}_multiscale"
    "${B}64${S}_multiscale_mid"
    "${B}64${S}_ms_l1_all"
    "${B}64${S}_ms_l1234_all"
    # job D -- encoder 64 level ablations
    "${B}64${S}_ms_l1_input"
    "${B}64${S}_ms_l2_mid"
    "${B}64${S}_ms_l12_mid"
    "${B}64${S}_ms_l34_mid"
    "${B}64${S}_ms_l14_mid"
)

OVERLAP=0.5              # must match the --overlap used when Pullback_prediction.py made the predictions
USE_POSTPROCESSED=true   # the stage-1 prediction jobs run with --postprocess
EVAL_RESOLUTION=encoder  # the stage-1 prediction jobs run with encoder resolution (matches the 3D U-Net)
LONG_ANGLE_STEP=15       # degrees between longitudinal rotation angles (0-165 deg)
PULLBACKS=""             # e.g. "NLD-RTG-0001" to only build one pullback (quick look); empty = all

overlap_suffix=$(echo "$OVERLAP" | tr -d '.')
eval_suffix=""
report_name="report.html"
if [ "$EVAL_RESOLUTION" = "encoder" ]; then
    eval_suffix="_encoderres"
    report_name="report_encoderres.html"
fi

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    NAME=$(basename "$MODEL_DIR")
    TM_DIR="$MODEL_DIR/test_metrics_${overlap_suffix}${eval_suffix}"
    REPORT="$TM_DIR/html_report/$report_name"

    # Pullback_prediction.py writes the report as its LAST step, so "report exists" == "this model's
    # prediction job has finished". Do NOT go by the predictions folder: it exists (partly filled)
    # while the job is still running, and the job would later overwrite our report anyway.
    if [ ! -f "$REPORT" ]; then
        echo ""
        echo "=== SKIP (prediction job not finished yet -- no report written): $NAME ==="
        continue
    fi
    if ! grep -q "const LONG_MANIFEST = \[\];" "$REPORT"; then
        echo ""
        echo "=== SKIP (report already has the longitudinal view): $NAME ==="
        continue
    fi

    echo ""
    echo "=== Building report: $NAME ==="
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
done

echo ""
echo "=== Batch complete ==="
