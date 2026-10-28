#!/bin/bash
# SIREN replicas of the ReLU encoder-width sweeps (enc64 SIREN runs already exist):
# baseline (no skip) and L1-skip, enc16/32/128/256. Run from repo root on the cluster:
#   bash Training_model/SBATCH_foregroundnorm/submit_all_siren_width_sweeps.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
for s in \
  encoder16_depth5_nodense_siren_zarrstore \
  encoder32_depth5_nodense_siren_zarrstore \
  encoder128_depth5_nodense_siren_zarrstore \
  encoder256_depth5_nodense_siren_zarrstore \
  encoder16_depth5_nodense_siren_zarrstore_multiscale_l1 \
  encoder32_depth5_nodense_siren_zarrstore_multiscale_l1 \
  encoder128_depth5_nodense_siren_zarrstore_multiscale_l1 \
  encoder256_depth5_nodense_siren_zarrstore_multiscale_l1 \
; do
  sbatch "$DIR/run_training_conditional_3D_cedice_${s}.sh"
done
