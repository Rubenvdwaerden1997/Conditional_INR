#!/bin/bash
# Submits the 3D U-Net NO-SKIP width sweep (16/32/64/128/256). Run from Training_model/UNet3D/.
for w in 16 32 64 128 256; do
    sbatch SBATCH/run_training_residualunet3d_encoder${w}_depth5_zarrstore_noskip.sh
done
