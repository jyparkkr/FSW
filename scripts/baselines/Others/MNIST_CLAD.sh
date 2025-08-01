#!/bin/bash

CURRENT="$PWD"
DATASET="MNIST" #MNIST FashionMNIST BiasedMNIST Drug Bios
TOKEN_LENGTH=0
PER_CLASS_EXAMPLE=1000000 # np.inf
VERBOSE=2
OPTIMIZER="sgd"
METHOD="CLAD"
METRIC="no_metrics"

for SEED in {0..4}; do
for EPOCH in 5; do
for TAU in 1.0; do
for LR in 0.001 0.01 0.1; do
for ETA in 1.0 2.0 4.0; do
for ALPHA in 0.0; do
for LAMBDA in 0.0; do
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
    elif [[ $DATASET == "Drug" ]]; then
        MODEL="MLP"
        NUM_TASK=3
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=32
    elif [[ $DATASET == "Bios" ]]; then
        MODEL="bert"
        NUM_TASK=5
        PER_TASK_CLASS=5
        BUFFER_PER_CLASS=32
    else
        ROOT="resnet18"
        NUM_TASK=10
        PER_TASK_CLASS=2
        BUFFER_PER_CLASS=64
    fi

    EXP_DUMP="dataset=${DATASET}"
    if [[ $TOKEN_LENGTH != 0 ]]; then
        EXP_DUMP="${EXP_DUMP}_${TOKEN_LENGTH}"
    fi
    EXP_DUMP="${EXP_DUMP}/${METHOD}_${ETA}"
    if [[ $OPTIMIZER != "sgd" ]]; then
        EXP_DUMP="${EXP_DUMP}_${OPTIMIZER}"
    fi
    EXP_DUMP="${EXP_DUMP}/${METRIC}"
    EXP_DUMP="${EXP_DUMP}/seed=${SEED}_epoch=${EPOCH}_lr=${LR}_tau=${TAU}"
    if [[ $ALPHA != 0.0 ]]; then
        EXP_DUMP="${EXP_DUMP}_alpha=${ALPHA}_optim=${OPTIM}"
    fi
    if [[ $LAMBDA != 0.0 ]]; then
        EXP_DUMP="${EXP_DUMP}_lmbd=${LAMBDA}_lmbdold=0.0"
    fi

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
                           --eta $ETA \
                           --optimizer $OPTIMIZER \
                           --learning_rate $LR \
                           --momentum 0.9 \
                           --learning_rate_decay 1.0 \
                           --metric $METRIC \
                           --fairness_agg "mean" \
                           --alpha $ALPHA \
                           --lambda $LAMBDA \
                           --lambda_old 0 \
                           --cuda 0 \
                           --verbose $VERBOSE \
                           1> $LOG_STDOUT 2> $LOG_STDERR
done
done
done
done
done
done
done