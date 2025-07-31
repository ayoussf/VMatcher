#!/bin/bash -l
SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../../"
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

hpatches_base_path=${PROJECT_DIR}/data/HPatches
python ./engine.py \
    --task=test \
    --config.test.dataset_settings.test-base-path=${hpatches_base_path}  \
    --config.test.dataset_settings.test-data-source=HPatches \
    --config.test.dataset_settings.test-data-root=${hpatches_base_path} \
    --config.test.dataset_settings.test-npz-root=None \
    --config.test.dataset_settings.test-list-path=None \
    --config.test.dataset_settings.test-intrinsic-path=None \
    --config.test.dataset_settings.ignore-scenes=True \
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
    --config.test.match-coarse.border-rm=0 \
    --config.test.metrics.homography-estimation-method=Poselib \
    --config.test.metrics.ransac-pixel-thr=3.0 \
    --config.test.metrics.topk=1024 \
    --config.test.plotting.enable-plotting=False \
    --config.test.mp=True \
    --config.test.fp32=False \
    --config.test.ckpt-path=