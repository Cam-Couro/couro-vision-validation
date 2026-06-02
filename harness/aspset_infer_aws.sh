#!/bin/bash
# ASPset-510 RTMPose inference on AWS g5.xlarge — same setup we used for OpenCap.
#
# Prerequisites locally:
#   - 22GB trainval-videos.tar.gz + 1.3GB test-videos.tar.gz downloaded
#   - AWS CLI configured with credentials that have EC2, ECR, S3, SSM access
#   - aws ec2 key pair available
#
# Steps:
#   1. Upload ASPset tar.gz files to S3 (~5-10 min on a typical connection)
#   2. Launch g5.xlarge with the right IAM role + security group
#   3. SSH/SSM into instance, pull Couro's prod ECR image, run inference
#   4. Push resulting keypoint JSONs to S3
#   5. Download keypoints locally (~50MB total)
#   6. Terminate instance
#
# Run this AFTER the local download completes.

set -euo pipefail

# -- CONFIGURATION ----------------------------------------------------------
DATA_DIR="/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/aspset510"
KEYPOINTS_DIR="/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/aspset510_couro_keypoints"

S3_BUCKET="${S3_BUCKET:-couro-research-tmp}"   # set this to a Couro-owned bucket
S3_INPUT_PREFIX="aspset510-inference/input"
S3_OUTPUT_PREFIX="aspset510-inference/output"

INSTANCE_TYPE="g5.xlarge"
AMI_ID="${AMI_ID:-ami-0123456789abcdef0}"      # AWS Deep Learning AMI Ubuntu, set per region
KEY_NAME="${KEY_NAME:-couro-research}"
IAM_PROFILE="${IAM_PROFILE:-couro-ec2-ssm-ecr}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-default}"
SUBNET_ID="${SUBNET_ID:-}"                      # leave blank to use default in us-east-1a/b/c

ECR_IMAGE="057663511299.dkr.ecr.us-east-1.amazonaws.com/rtm-pose-inference-prod:latest"

# Estimated cost: ~$1/hr × 3-4 hours = $3-5 in compute
# -------------------------------------------------------------------------

echo "=== Step 1: Upload ASPset tar.gz to S3 ==="
aws s3 cp "$DATA_DIR/aspset510_v1_trainval-videos.tar.gz" \
    "s3://$S3_BUCKET/$S3_INPUT_PREFIX/" --no-progress
aws s3 cp "$DATA_DIR/aspset510_v1_test-videos.tar.gz" \
    "s3://$S3_BUCKET/$S3_INPUT_PREFIX/" --no-progress
echo "Done."

echo ""
echo "=== Step 2: Launch g5.xlarge instance ==="
USER_DATA=$(cat <<'EOF'
#!/bin/bash
# Hard kill after 4 hours so a stuck instance doesn't burn money
( sleep 14400 && shutdown -h now ) &
exec >> /var/log/user-data.log 2>&1
set -ex
apt-get update
apt-get install -y awscli unzip
EOF
)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    ${SUBNET_ID:+--subnet-id "$SUBNET_ID"} \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aspset-rtmpose-inference},{Key=Project,Value=couro-research}]' \
    --query 'Instances[0].InstanceId' --output text)
echo "Launched: $INSTANCE_ID"

echo "Waiting for instance to be SSM-ready (typically 2-3 min)..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
sleep 60  # give SSM agent time to register
# Wait for SSM
for i in {1..20}; do
    if aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].InstanceId' --output text 2>/dev/null | grep -q "$INSTANCE_ID"; then
        echo "SSM ready."
        break
    fi
    sleep 15
done

echo ""
echo "=== Step 3: Run inference inside instance via SSM ==="
# This sends a shell script to the EC2 instance that pulls the ECR image,
# downloads the videos from S3, runs inference, and pushes results back to S3.
cmd=$(cat <<EOF
set -ex
mkdir -p /home/ubuntu/work && cd /home/ubuntu/work
aws s3 cp s3://$S3_BUCKET/$S3_INPUT_PREFIX/aspset510_v1_trainval-videos.tar.gz .
tar xzf aspset510_v1_trainval-videos.tar.gz
rm aspset510_v1_trainval-videos.tar.gz

# Login to ECR + pull image
aws ecr get-login-password --region us-east-1 | docker login --username AWS \\
    --password-stdin 057663511299.dkr.ecr.us-east-1.amazonaws.com
docker pull $ECR_IMAGE

# Pre-download Couro model weights so the container doesn't re-fetch each time
# (Same trick we used for OpenCap)
mkdir -p model
aws s3 cp --recursive s3://couro-model-weights/prod/ model/ || true

# Run the ASPset-specific driver
docker run --gpus all -v /home/ubuntu/work:/work --rm $ECR_IMAGE \\
    python /work/aspset_infer.py /work/ASPset-510/trainval/videos /work/keypoints_out

# Upload results
aws s3 cp --recursive keypoints_out/ s3://$S3_BUCKET/$S3_OUTPUT_PREFIX/

echo INFERENCE_DONE
EOF
)

aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$cmd\"]" \
    --query 'Command.CommandId' --output text > /tmp/cmd_id.txt

CMD_ID=$(cat /tmp/cmd_id.txt)
echo "SSM command sent: $CMD_ID"
echo "Monitor with: aws ssm get-command-invocation --command-id $CMD_ID --instance-id $INSTANCE_ID"

echo ""
echo "=== Step 4: Wait for inference (~3-4 hours), then pull keypoints ==="
echo "When you see INFERENCE_DONE in the SSM output:"
echo "  mkdir -p $KEYPOINTS_DIR"
echo "  aws s3 cp --recursive s3://$S3_BUCKET/$S3_OUTPUT_PREFIX/ $KEYPOINTS_DIR/"
echo ""
echo "=== Step 5: Terminate instance ==="
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
