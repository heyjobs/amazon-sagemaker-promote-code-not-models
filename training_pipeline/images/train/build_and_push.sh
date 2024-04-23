# read profile-id from config-file
config_file="profiles.conf"
source "$config_file"
#echo "Profile-Id: $operations"
operations=676012288735
echo "Profile-Name: operations"

# Login to AWS docker registry
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin $operations.dkr.ecr.eu-central-1.amazonaws.com

# build docker image (if running on M1/M2 --> specify platform with : --platform linux/amd64)
docker build -t training-image -f images/train/Dockerfile .

# Login to docker registry
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin $operations.dkr.ecr.eu-central-1.amazonaws.com

# Tag and Push Docker Image to Container Registry
docker tag training-image $operations.dkr.ecr.eu-central-1.amazonaws.com/training-image:latest
docker push $operations.dkr.ecr.eu-central-1.amazonaws.com/training-image:latest