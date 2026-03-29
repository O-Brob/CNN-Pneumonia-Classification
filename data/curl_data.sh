#!/bin/bash

# Download and unzip Kaggle dataset
curl -L -o ./chest-xray-pneumonia.zip\
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia

unzip ./chest-xray-pneumonia.zip

# Create output directories for both Pneumonia & Normal x-rays
mkdir ./Pneumonia
mkDir ./Normal

# Gather all Pneumonia x-rays in ./Pneumonia and
# all Normal x-rays in ./Normal, to allow manual split.
mv ./chest_xray/*/Pneumonia/* ./Pneumonia/
mv ./chest_xray/*/Normal/* ./Normal/

# Clean up the no longer needed files.
rm -rf ./chest_xray
rm -rf ./chest-xray-pneumonia.zip