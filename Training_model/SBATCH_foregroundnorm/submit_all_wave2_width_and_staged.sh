#!/bin/bash
# Wave 2: layer-1-at-mid at encoder 128/256 (qos=low / qos=vram) + staged injection at encoder 64
# (coarse->fine = U-Net order, fine->coarse = reverse control).
# Run from the repo root on the cluster so ./SLURM/Slurm_output_3D resolves:
#   bash Training_model/SBATCH_foregroundnorm/submit_all_wave2_width_and_staged.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder128_depth5_nodense_relu_zarrstore_multiscale_l1_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder256_depth5_nodense_relu_zarrstore_multiscale_l1_mid.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_staged_c2f.sh"
sbatch "$DIR/run_training_conditional_3D_cedice_encoder64_depth5_nodense_relu_zarrstore_ms_staged_f2c.sh"
