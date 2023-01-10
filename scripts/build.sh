#!/bin/bash

# Auto setup submodule on building (good for CI)

# Auto compile protocol buffer code as well
bash ./scripts/compile-protoc.sh

rm -rf ./build || true
mkdir ./build

cp -r ./requirements.txt ./src/* ./build/
mkdir ./build/service/tessdata_best
cp -r ./tessdata_best ./build/service/
TESSDATA_PREFIX="$(pwd)/service/tessdata_best"
