#!/usr/bin/env bash

echo "========================================================================"
echo "pip install pytorch:"
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

echo "========================================================================"
echo "build our revised mmcv:"
cd lib/mmcv
python setup.py clean
rm -rf build
MMCV_WITH_OPS=1 pip install -e .
cd ../../

echo "========================================================================"
echo "pip install some other packages:"
pip install -r requirements.txt


