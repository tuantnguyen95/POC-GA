#!/bin/bash

modelPath="$1"

docker run -t --rm -p 8500:8500 -v "$modelPath:/models/element_embeddings" -e MODEL_NAME=element_embeddings tensorflow/serving:1.12.3
