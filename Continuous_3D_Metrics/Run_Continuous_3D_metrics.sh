#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./slurm_output/_slurm_output_continuous3d_%j.txt
#SBATCH -e ./slurm_output/_slurm_error_continuous3d_%j.txt
#SBATCH --qos=high
#SBATCH --exclude=dlc-mewtwo,dlc-moltres,dlc-nidoking,dlc-scyther,dlc-lugia,dlc-zapdos

cd /data/diag/rubenvdw/Conditional_INR/Continuous_3D_Metrics

# No GPU needed here -- this script only reads already-written prediction .nii.gz files
# and runs CPU-side numpy/plaque_quantification, no model inference.
# pandas/openpyxl -> writing the output .xlsx; SimpleITK -> reading prediction .nii.gz;
# scipy/Pillow -> plaque_quantification.py's arc/FCT/calcium-depth geometry + font rendering.
pip install -U pandas openpyxl SimpleITK scipy pillow

echo "Running on node: $(hostname)"

# All four folders verified (dry run) to resolve to 29/29 pullbacks each, matching
# Test_set_Performance/Dicoms -- see order below, PRED_DIRS/MODEL_NAMES/LABEL_FILES stay parallel.
PRED_DIRS=(
    "/data/diag/rubenvdw/Test_set_Performance/Segmentation_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense_foregroundnorm"
    "/data/diag/rubenvdw/Test_set_Performance/Segmentation_OCTAID"
    "/data/diag/rubenvdw/Test_set_Performance/Segmentation_OCTAIDlite/_student"
    "/data/diag/rubenvdw/Test_set_Performance/Segmentation_UNET3D"
)
MODEL_NAMES=(
    "ConditionalINR"
    "OCTAID"
    "OCTAIDlite"
    "UNET3D"
)
# Per-folder label file (parallel to PRED_DIRS) -- only OCTAIDlite's 14-class taxonomy
# differs from canonical, so only it needs one; "" means "already canonical".
LABEL_FILES=(
    "/data/diag/rubenvdw/Conditional_INR/Continuous_3D_Metrics/label_description_octaid_conditional_unet3d.txt"
    "/data/diag/rubenvdw/Conditional_INR/Continuous_3D_Metrics/label_description_octaid_conditional_unet3d.txt"
    "/data/diag/rubenvdw/Conditional_INR/Continuous_3D_Metrics/label_description_octaidlite.txt"
    "/data/diag/rubenvdw/Conditional_INR/Continuous_3D_Metrics/label_description_octaid_conditional_unet3d.txt"
)

# Optional: excel with 'pullback', 'guiding', 'artifact' columns (frame ranges, e.g. "12-100").
# Leave empty to skip exclusion handling entirely -- not yet provided as of this run.
EXCLUSION_EXCEL="/data/diag/rubenvdw/Conditional_INR/Continuous_3D_Metrics/Exclusion_frames_testset.xlsx"

# Confirmed by smoke test (2026-08-05): only ConditionalINR's own writer sets correct
# .nii.gz spacing; OCTAID/OCTAIDlite/UNET3D all carry the SimpleITK default identity
# spacing (1,1,1)mm, which silently inflated FCT/area/length by ~100x if trusted.
# XY_SPACING_MM is a genuine fixed constant (catheter/console optical property, confirmed
# via the pullbacks' own DICOM PixelSpacing=0.009957325mm, matching plaque_quantification.py's
# own hardcoded default) -- forced for every folder regardless of what each header claims.
XY_SPACING_MM=0.009957325

# Z spacing is deliberately NOT forced to one constant (2026-08-06 correction) -- unlike
# xy, it genuinely varies by pullback-length protocol (75mm pullbacks run at 350-400
# frames -> ~0.2mm/frame; 54mm pullbacks at 250-300 or 500-550 frames -> ~0.1-0.2mm/frame).
# A single fixed 0.10mm (this project's Training_model/config.py default) was verified wrong
# by ~2x for the 375/374/269/270-frame pullbacks in this test set. Leaving Z_SPACING_MM empty
# lets Continuous_3D_metrics.py's infer_z_spacing_mm() pick the right value per pullback from
# its own frame count instead -- only set this if you want to force one value for every
# pullback regardless of its actual protocol.
Z_SPACING_MM=""

TCFA_FCT_THRESHOLD_UM=65.0
TCFA_LIPID_ARC_THRESHOLD_DEG=90.0
TCFA_MIN_CONSECUTIVE_FRAMES=1   # 1 = no continuity filter; every run (incl. single frames) kept in TCFA_Lesions

# Verified 2026-08-05 (parallel smoke test, n_procs=4 on 8 jobs): output is byte-identical
# to the sequential run for the same pullback, and wall time drops roughly in proportion to
# --n_procs (jobs are fully independent, no shared state). Match --n_procs to --cpus-per-task
# above. 116 total (model, pullback) jobs (29 pullbacks x 4 models); sequential would be
# ~20-25h based on the smoke test's per-job timings, so --n_procs=16 should land in the
# 1.5-2h range rather than overnight+.
N_PROCS=16

ARGS=(
    --pred_dirs "${PRED_DIRS[@]}"
    --model_names "${MODEL_NAMES[@]}"
    --label_files "${LABEL_FILES[@]}"
    --env cluster
    --xy_spacing_mm "$XY_SPACING_MM"
    --tcfa_fct_threshold_um "$TCFA_FCT_THRESHOLD_UM"
    --tcfa_lipid_arc_threshold_deg "$TCFA_LIPID_ARC_THRESHOLD_DEG"
    --tcfa_min_consecutive_frames "$TCFA_MIN_CONSECUTIVE_FRAMES"
    --n_procs "$N_PROCS"
)
if [ -n "$Z_SPACING_MM" ]; then
    ARGS+=(--z_spacing_mm "$Z_SPACING_MM")
fi
if [ -n "$EXCLUSION_EXCEL" ]; then
    ARGS+=(--exclusion_excel "$EXCLUSION_EXCEL")
fi

python3 -u Continuous_3D_metrics.py "${ARGS[@]}"
