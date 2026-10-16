#!/bin/bash
# Resubmits the three 2D runs that had not finished (checked 2026-10-14 from train.log):
#   INR    encoder64  -- stopped at epoch 256/300
#   INR    encoder256 -- stopped at epoch 243/300
#   U-Net  encoder32  -- stopped at epoch 185/300
# The other seven 2D runs completed all 300 epochs and are not resubmitted.
# Both main.py and train_unet.py resume from latest.pt in the run's save dir (saved every epoch),
# so these continue from where they stopped and append to the existing train.log.
# Before running, check on the cluster (squeue -u $USER) that none of the three is still running
# or already resubmitted -- the W: share can lag, and a duplicate job would write to the same dir.
#
# Not an sbatch script itself: run it on the login node from anywhere, e.g.
#     bash /data/diag/rubenvdw/Conditional_INR/Training_model/submit_all_2d.sh
# Log paths in the scripts are relative, so we cd to Training_model first.

cd /data/diag/rubenvdw/Conditional_INR/Training_model || exit 1
mkdir -p SLURM/Slurm_output_3D

sbatch "ZARR_format/SBATCH/run_training_conditional_3D_cedice_encoder64_depth5_2d_foregroundnorm.sh"
sbatch "ZARR_format/SBATCH/run_training_conditional_3D_cedice_encoder256_depth5_2d_foregroundnorm.sh"
sbatch "UNet2D/SBATCH/run_training_residualunet2d_encoder32_depth5_zarrstore_skipconnections.sh"
