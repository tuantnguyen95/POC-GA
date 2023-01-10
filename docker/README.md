# Dockerinze booster-ai-service

## How to build docker image

Follow below steps

1. Install [docker](https://docs.docker.com/install/) (do once)
1. Locate at root dir, run `yarn dockerize`
1. Done

For any customization, please getting start from script `dockerize` in `package.json`

## Deploy on GPU instance
1. Install GPU driver. Check out this [guideline](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html) from AWS for what version we need for each type of instance. Ex: Tesla T4 needs version 418 or later.
2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start): Keep in mind that you need docker `19.3` or later which will let you use the `--gpus` option (`--runtime=nvidia` is deprecated).
3. Pull latest version of serving gpu: 

```bash
docker pull tensorflow/serving:latest-gpu
```

4. Run the docker as we usually did with the serving cpu + plus the `--gpus` option.

For more details, please check out:

1. [Serving with docker using gpu](https://www.tensorflow.org/tfx/serving/docker#serving_with_docker_using_your_gpu) 
2. [Developing with Docker guide.](https://www.tensorflow.org/tfx/serving/building_with_docker)