#!/bin/bash

CURRENT="$PWD"
DATASET="BiasedMNIST" #MNIST FashionMNIST BiasedMNIST
PER_CLASS_EXAMPLE=100000 # np.inf
TAU=5
ALPHA=0.002
LAMBDA=0.01
VERBOSE=2
METHOD="FSW"
METRIC="DP"

cnt=0
# for SEED in {0..4}; do
for SEED in 0 1; do
for EPOCH in 15; do
for TAU in 5.0 10.0; do
for LR in 0.001; do
for ALPHA in 0.005 0.01 0.002; do
for LAMBDA in 0.1 0.2 0.4 0.5 0.6 0.8 1.0 2.0 3.0 4.0 5.0 7.0 9.0 10.0; do
# for LAMBDA in 0.5; do
    if [[ $DATASET == "MNIST" ]]; then
        MODEL="MLP"
        NUM_TASK=5
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=32
    elif [[ $DATASET == "FashionMNIST" ]]; then
        MODEL="MLP"
        NUM_TASK=5
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=32
    elif [[ $DATASET == "BiasedMNIST" ]]; then
        MODEL="MLP"
        NUM_TASK=5
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=64
    elif [[ $DATASET == "CIFAR10" ]]; then
        ROOT="resnet18"
        NUM_TASK=5
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=32
    else
        ROOT="resnet18"
        NUM_TASK=10
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=32
    fi

    EXP_DUMP="dataset=${DATASET}/${METHOD}/${METRIC}"
    EXP_DUMP="${EXP_DUMP}/seed=${SEED}_epoch=${EPOCH}_lr=${LR}_tau=${TAU}"
    if [[ $ALPHA != 0.0 ]]; then
        EXP_DUMP="${EXP_DUMP}_alpha=${ALPHA}"
    fi
    if [[ $LAMBDA != 0.0 ]]; then
        EXP_DUMP="${EXP_DUMP}_lmbd=${LAMBDA}_lmbdold=0.0"
    fi

    echo "EXP_DUMP:"
    echo "$EXP_DUMP" > /dev/stdout

    OUT_FOLDER="scripts_output_rebuttal/${EXP_DUMP}"
    LOG_STDOUT="${OUT_FOLDER}/log.out"
    LOG_STDERR="${OUT_FOLDER}/log.err"

    echo "Waiting for 5 seconds..." > /dev/stdout
    sleep 4
    sleep 1
    echo "Task Start"  > /dev/stdout
    mkdir -p $OUT_FOLDER
    ~/anaconda3/envs/cil/bin/python run.py \
                           --rebuttal \
                           --dataset $DATASET \
                           --model $MODEL \
                           --method $METHOD \
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
                           --metric $METRIC \
                           --fairness_agg "mean" \
                           --alpha $ALPHA \
                           --lambda $LAMBDA \
                           --lambda_old 0 \
                           --cuda 7 \
                           --verbose $VERBOSE \
                        #    1> $LOG_STDOUT 2> $LOG_STDERR
done
done
done
done
done
done