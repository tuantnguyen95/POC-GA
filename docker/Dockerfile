# The order of statements are optimized for docker cache
# Be caution when changing their order
FROM python:3.8

LABEL author="Kobiton, Inc."
LABEL website="https://kobiton.com"

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y tesseract-ocr ffmpeg libsm6 libxext6

# Below envs required at launching
ENV KOBITON_AI_TF_SERVING_HOSTNAME 0.0.0.0
ENV KOBITON_AI_SERVICE_HOSTNAME 0.0.0.0

# Below envs optionals at launching
ENV KOBITON_AI_TF_SERVING_PORT 8500
ENV KOBITON_AI_SERVICE_PORT 5000
ENV DEPLOY_ENV "test"

#Disable tesseract multithreading
ENV OMP_THREAD_LIMIT=1

# The executiont time for "pip3 install" is pretty long, so copy "requirements.txt" file
# to utilize build cache
COPY ./build/requirements.txt .
RUN pip3 install Cython --install-option="--no-cython-compile"
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./build/ .

CMD python -m service.app \
  $KOBITON_AI_TF_SERVING_HOSTNAME $KOBITON_AI_TF_SERVING_PORT \
  $KOBITON_AI_SERVICE_HOSTNAME $KOBITON_AI_SERVICE_PORT \
  --logstash_host $LOGSTASH_HOST \
  --deploy_env $DEPLOY_ENV
