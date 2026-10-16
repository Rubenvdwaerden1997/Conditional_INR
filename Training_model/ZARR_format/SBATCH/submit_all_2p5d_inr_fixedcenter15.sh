#!/bin/bash
# Submit the 5 fixed-centre 2.5D conditional-INR runs (encoder 16/32/64/128/256).
# Run from the repo root on the cluster so ./SLURM/Slurm_output_3D resolves:
#   bash Training_model/ZARR_format/SBATCH/submit_all_2p5d_inr_fixedcenter15.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
for w in 16 32 64 128 256; do
  sbatch "$DIR/run_training_conditional_3D_cedice_encoder${w}_depth5_2p5d_foregroundnorm_fixedcenter15.sh"
done
