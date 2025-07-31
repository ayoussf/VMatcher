#!/bin/bash -l
SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
scannet_base_path=${PROJECT_DIR}/assets/scannet_test_1500
# Array of pixel threshold values
thresholds=(0.5 1.0 1.5 2.0 2.5 3.0)

# Loop over each threshold value and run the script with that value
for thr in "${thresholds[@]}"
do
  echo "Running with ransac-pixel-thr=${thr}"
  
  python ./engine.py \
    --task=test \
    --config.test.dataset_settings.test-base-path=scannet_base_path \
    --config.test.dataset_settings.test-data-source=ScanNet \
    --config.test.dataset_settings.test-data-root=${PROJECT_DIR}/data/scannet/test/scannet_test_1500 \
    --config.test.dataset_settings.test-npz-root=${scannet_base_path} \
    --config.test.dataset_settings.test-list-path=${scannet_base_path}/scannet_test.txt \
    --config.test.dataset_settings.test-intrinsic-path=${scannet_base_path}/intrinsics.npz \
    --config.test.backbone.backbone-type=VGG \
    --config.test.mamba-config.model-type=mamba_v \
    --config.test.mamba-config.num-layers=24 \
    --config.test.mamba-config.switch=True \
    --config.test.mamba-config.bidirectional=False \
    --config.test.mamba-config.divide-output=False \
    --config.test.mamba-config.residual-in-fp32=False \
    --config.test.atten-config.downsample=True \
    --config.test.atten-config.flash=False \
    --config.test.match-coarse.thr=0.05 \
    --config.test.metrics.pose-estimation-method=LO-RANSAC \
    --config.test.metrics.ransac-pixel-thr=${thr}\
    --config.test.metrics.epi-err-thr=0.0005 \
    --config.test.plotting.enable-plotting=False \
    --config.test.mp=True \
    --config.test.fp32=False \
    --config.test.ckpt-path=
done