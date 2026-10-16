#!/bin/bash
# Submits the 3D INR-size sweeps as separate SLURM jobs (10 jobs):
#   3D INR, encoder FIXED at feature_size=32 (enc32) or 16 (enc16), inr_hidden in 8 16 32 64 256
#   (inr_hidden=128 is the existing baseline at each encoder size -- enc32: Val Dice 0.5745,
#   enc16: 0.4834 -- not resubmitted).
# Compared against the existing 3D U-Net width sweep (16/32/64/128/256) on a
# Val-Dice-vs-total-parameters curve. Total INR params (local build):
#   enc32: 102k 104k 109k 125k [174k] 347k      enc16: 27k 28k 33k 46k [92k] 256k
#
# Not an sbatch script itself: run it on the login node from anywhere, e.g.
#     bash /data/diag/rubenvdw/Conditional_INR/Training_model/submit_all_3d.sh
#
# PRIORITY: set HIGH_PRIORITY below to the run that should get --qos=high. It is passed on the
# sbatch command line, so it overrides the #SBATCH --qos line inside that run's own script.
# Valid values: e32_h8 e32_h16 e32_h32 e32_h64 e32_h256 e16_h8 e16_h16 e16_h32 e16_h64 e16_h256
# (empty = none). NOTE: e32_h256 is submitted with --qos=vram in its own script (wider MLP
# over 524288 points); making it high replaces vram, so only pick it if that is what you want.
# enc16 runs use --qos=low, except e16_h256 (vram, see SKIP_TAGS note below).
HIGH_PRIORITY=""

# ALREADY SUBMITTED -- skipped so rerunning this script never creates duplicate jobs. Two jobs on
# the same run would share one save dir and both resume from / overwrite the same latest.pt.
# enc32 (all five) and enc16 h8/h16/h32/h64 finished 2026-10-15. e16_h256 is NOT skipped: it OOMed on a
# low-QoS GPU (10.6 GiB) on 2026-10-14, so its script now uses --qos=vram and it must be resubmitted.
# Remove a tag from this list only if that run was cancelled or failed and must be resubmitted.
SKIP_TAGS=" e32_h8 e32_h16 e32_h32 e32_h64 e32_h256 e16_h8 e16_h16 e16_h32 e16_h64 "

cd /data/diag/rubenvdw/Conditional_INR/Training_model || exit 1
mkdir -p SLURM/Slurm_output_3D

submit() {   # submit <tag> <script>
    local tag="$1" script="$2"
    if [[ "$SKIP_TAGS" == *" $tag "* ]]; then
        echo "[$tag] skipped (already submitted)"
        return
    fi
    # Second guard: skip if a job with this script's name is already pending/running for this user
    # (sbatch names a job after its script file when no --job-name is set).
    if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$(basename "$script")"; then
        echo "[$tag] skipped (already in the queue)"
        return
    fi
    if [ "$tag" = "$HIGH_PRIORITY" ]; then
        echo "[$tag] qos=high"
        sbatch --qos=high "$script"
    else
        echo "[$tag]"
        sbatch "$script"
    fi
}

for enc in 32 16; do
    for h in 8 16 32 64 256; do
        submit "e${enc}_h${h}" "ZARR_format/SBATCH/run_training_conditional_3D_cedice_encoder${enc}_depth5_nodense_relu_zarrstore_inrhidden${h}.sh"
    done
done
