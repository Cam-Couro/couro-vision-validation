#!/bin/bash
# Run this AFTER AWS inference completes (monitor will signal DONE)
set -euo pipefail

DATA_DIR=/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data
KP_DIR=$DATA_DIR/aspset510_couro_keypoints
RESULTS_BUCKET=couropt-pretrained-models-us-east-1
INSTANCE=i-03025dec51c68c649

echo "=== Step 1: Pull keypoint JSONs from S3 ==="
mkdir -p "$KP_DIR"
aws s3 sync s3://$RESULTS_BUCKET/aspset_keypoints/ "$KP_DIR/"
echo "Got $(ls $KP_DIR/*.json | wc -l) keypoint files"

echo ""
echo "=== Step 2: Terminate AWS instance ==="
aws ec2 terminate-instances --instance-ids $INSTANCE --query 'TerminatingInstances[0].CurrentState.Name' --output text

echo ""
echo "=== Step 3: Run combined OpenCap + ASPset training ==="
cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
python3 -m harness.train_v12_combined 2>&1 | tee /tmp/v12_combined.log | grep -v RuntimeWarning | grep -v "warnings.warn"

echo ""
echo "=== Step 4: Re-finalize deploy_ready_models with v12 included ==="
# (we'll wire v12 into finalize_best_models if results look promising)

echo ""
echo "=== DONE. Results in: ==="
ls -la /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results/v12_combined_models.json
