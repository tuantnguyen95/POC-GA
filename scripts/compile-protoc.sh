#!/bin/bash

rootDir=`dirname $(dirname $0)`

cd $rootDir/schema
rm -rf ../src/service/schema &>/dev/null || true
yarn compile ../src/service/schema
yarn compile-python ../src/service/schema
