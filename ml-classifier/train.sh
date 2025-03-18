#!/bin/bash

wd=$(pwd)
mkdir data
mkdir $wd/output

unzip train.zip -d data/
rm train.zip
# unzip test.zip -d data/
# rm test.zip

cd /app/
python train.py \
  --data-dir $wd/data/ \
  --checkpoint-dir $wd/output/

