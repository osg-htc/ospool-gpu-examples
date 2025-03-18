#!/bin/bash

wd=$(pwd)
mkdir data
mkdir $wd/output

unzip test.zip
rm test.zip

python /app/infer.py \
  --data-dir $wd/test/ \
  --model-path $wd/model.pth

