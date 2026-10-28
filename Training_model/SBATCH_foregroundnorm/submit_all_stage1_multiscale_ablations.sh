#!/bin/bash
# Stage-1 multiscale ablations (3D INR, ReLU): level subsets, entry point, 'all layers', and small widths.
# Run from the repo root on the cluster so ./SLURM/Slurm_output_3D resolves:
#   bash Training_model/SBATCH_foregroundnorm/submit_all_stage1_multiscale_ablations.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l2_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l12_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l34_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l1_input.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l14_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l1_all.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_l1234_all.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder16_depth5_nodense_relu_zarrstore_multiscale_l1_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder32_depth5_nodense_relu_zarrstore_multiscale_l1_mid.sh"
