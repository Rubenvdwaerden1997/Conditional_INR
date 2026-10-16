#!/bin/bash
# Submit the 5 encoder-64 3D INR runs (multiscale at input / mid / layer1-only mid, plus SIREN).
# Run from the repo root on the cluster so ./SLURM/Slurm_output_3D resolves:
#   bash Training_model/SBATCH_foregroundnorm/submit_all_encoder64_multiscale_siren.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
for s in multiscale multiscale_mid multiscale_l1_mid; do
  sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_${s}.sh"
done
for s in siren_zarrstore siren_zarrstore_multiscale_l1; do
  sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_${s}.sh"
done
