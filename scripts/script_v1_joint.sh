#!/bin/bash

CURRENT="$PWD"
DATASET="FashionMNIST" #MNIST FashionMNIST BiasedMNIST
PER_CLASS_EXAMPLE=100000 # np.inf
TAU=5
ALPHA=0.00
LAMBDA=1.0
VERBOSE=2

cnt=0
for SEED in {0..4}; do
# for SEED in 10; do
for EPOCH in 1 5; do
for TAU in 0.0; do
for LR in 0.001 0.01; do
for ALPHA in 0.0; do
for DATASET in "FashionMNIST" "MNIST"; do
for LAMBDA in 0.0; do
    if [[ $DATASET == "MNIST" ]]; then
        MODEL="MLP"
        NUM_TASK=1
        PER_TASK_CLASS=2
        PER_CLASS_EXAMPLE=5000
        BUFFER_PER_CLASS=32
        METRIC="std"
    elif [[ $DATASET == "FashionMNIST" ]]; then
        MODEL="MLP"
        NUM_TASK=1
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=32
        METRIC="std"
    elif [[ $DATASET == "BiasedMNIST" ]]; then
        MODEL="MLP"
        NUM_TASK=1
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=64
        METRIC="EO"
    else
        ROOT="resnet18"
        NUM_TASK=1
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=64
        METRIC="std"
    fi

    EXP_DUMP="dataset=${DATASET}/joint/seed=${SEED}_epoch=${EPOCH}_lr=${LR}"
    echo "EXP_DUMP:"
    echo "$EXP_DUMP" > /dev/stdout

    OUT_FOLDER="scripts_output/${EXP_DUMP}"
    LOG_STDOUT="${OUT_FOLDER}/log.out"
    LOG_STDERR="${OUT_FOLDER}/log.err"

    echo "Waiting for 5 seconds..." > /dev/stdout
    sleep 4
    sleep 1
    echo "Task Start"  > /dev/stdout
    mkdir -p $OUT_FOLDER
    ~/anaconda3/envs/cil/bin/python run.py \
                           --dataset $DATASET \
                           --model $MODEL \
                           --seed $SEED \
                           --num_task $NUM_TASK \
                           --epochs_per_task $EPOCH \
                           --per_task_examples $(($PER_CLASS_EXAMPLE * $PER_TASK_CLASS)) \
                           --per_task_memory_examples $(($BUFFER_PER_CLASS * $PER_TASK_CLASS)) \
                           --batch_size_train 64 \
                           --batch_size_memory 64 \
                           --batch_size_validation 256 \
                           --tau $TAU \
                           --optimizer sgd \
                           --learning_rate $LR \
                           --momentum 0.9 \
                           --learning_rate_decay 1.0 \
                           --algorithm optimization \
                           --metric $METRIC \
                           --fairness_agg "mean" \
                           --alpha $ALPHA \
                           --lambda $LAMBDA \
                           --lambda_old 0 \
                           --cuda 6 \
                           --verbose $VERBOSE \
                           1> $LOG_STDOUT 2> $LOG_STDERR
done
done
done
done
done
done
done