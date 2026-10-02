#!/bin/bash
# Build and push the dedicated training image for Conditional_INR. Run this yourself on a
# machine with docker (and registry push access) -- not something to submit via sbatch, and not
# something run automatically by any training script in this repo.
#
# Usage:
#   cd Dockerfile_initiation
#   bash build_and_push.sh v1.7        # pick the next tag -- v1.6 is already in use/working
#                                       # (verified on both dlc-jolteon and an older GPU node,
#                                       # rolled out to all real training scripts 2026-10-02),
#                                       # don't overwrite it until a new tag is itself verified --
#                                       # see STEPS.txt step 7
set -euo pipefail

TAG="${1:?usage: bash build_and_push.sh <tag, e.g. v1.5>}"
IMAGE="dockerdex.umcn.nl:5005/rubenvdwaerden1997/train_monai:${TAG}"

docker build -t "$IMAGE" .
docker push "$IMAGE"

echo ""
echo "Pushed $IMAGE"
echo "Before pointing any real training .sh at this tag, verify it works: update one script's"
echo "--container-image to this tag, drop its pip3 install line entirely, and rerun that script's"
echo "interactive_test.sh --sanity_check first, then a short sbatch smoke-train."
