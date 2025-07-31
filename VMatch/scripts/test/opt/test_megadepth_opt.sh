#!/bin/bash -l
SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../../"
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

python ./engine.py \
    --task=test \
    --config.test.backbone.backbone-type=VGG \
    --config.test.mamba-config.model-type=mamba_v \
    --config.test.mamba-config.num-layers=24 \
    --config.test.mamba-config.switch=True \
    --config.test.mamba-config.bidirectional=False \
    --config.test.mamba-config.divide-output=False \
    --config.test.mamba-config.residual-in-fp32=False \
    --config.test.atten-config.downsample=True \
    --config.test.atten-config.flash=False \
    --config.test.match-coarse.thr=10 \
    --config.test.match-coarse.skip-softmax=True \
    --config.test.match-coarse.fp16matmul=True \
    --config.test.metrics.pose-estimation-method=LO-RANSAC \
    --config.test.metrics.ransac-pixel-thr=2.0 \
    --config.test.dataset-settings.mgdpt-img-resize=1184 \
    --config.test.plotting.enable-plotting=False \
    --config.test.mp=True \
    --config.test.fp32=False \
    --config.test.ckpt-path=