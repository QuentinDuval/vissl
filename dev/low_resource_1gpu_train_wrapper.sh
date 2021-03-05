#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# ----------------------------------- USAGE ----------------------------------------------------- #
# TYPE=engine ./dev/low_resource_1gpu_train_wrapper.sh config=pretrain/swav/swav_8node_resnet
# ----------------------------------------------------------------------------------------------- #

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$(dirname "${SRC_DIR}")
CHECKPOINT_DIR=$(mktemp -d)
TYPE=${TYPE-engine}   # engine | svm | cluster | knn
CFG=( "$@" )
DEVICE=${DEVICE-gpu}
DIST_BACKEND=${DIST_BACKEND-nccl}

########################## Select the binary ##################################
if [ "$TYPE" = "engine" ]; then
  BNAME=run_distributed_engines
elif [ "$TYPE" = "cluster" ]; then
  BNAME=cluster_features_and_label
elif [ "$TYPE" = "knn" ]; then
  BNAME=nearest_neighbor_test
else
  BNAME=train_svm
fi
BINARY="python ${SRC_DIR}/tools/${BNAME}.py"

echo "========================================================================"
echo "SRC_DIR: $SRC_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "Setting to run: "
echo "${CFG[@]}"
echo "========================================================================"

echo "Starting...."
# shellcheck disable=SC2102
# shellcheck disable=SC2086
$BINARY ${CFG[*]} \
    config.DATA.NUM_DATALOADER_WORKERS=0 \
    config.MACHINE.DEVICE=$DEVICE \
    config.MULTI_PROCESSING_METHOD=forkserver \
    config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.RUN_ID=auto \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.DISTRIBUTED.BACKEND=$DIST_BACKEND \
    config.CHECKPOINT.DIR="$CHECKPOINT_DIR" \
    config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=16 \
    config.DATA.TRAIN.DATA_LIMIT.NUM_SAMPLES=500 \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=16 \
    config.DATA.TEST.DATA_LIMIT.NUM_SAMPLES=500
