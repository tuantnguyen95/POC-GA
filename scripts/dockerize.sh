#!/bin/bash

# This script builds the docker image for booster-ai-service
# Note: the outcome docker image doesn't include the trained model which is served separately by
# tensorflow/serving image

# These params are for CI to control the output docker image
docker_image_name="${KOBITON_CI_DOCKER_IMAGE_NAME:-booster-ai-service}"
docker_image_tag="${KOBITON_CI_DOCKER_IMAGE_TAG:-latest}"

cd docker

# Prepare the build folder
rm -rf ./build || true
mkdir ./build

cp -r ../build/* ./build/

commitId=`git rev-parse HEAD | cut -c 1-10`

docker build  -t $docker_image_name:$docker_image_tag --label "commit.id=$commitId" --label "built.at=$(date -u)" .

