#!/bin/bash

NUM_TASK=5
PER_TASK_CLASS=2
PER_CLASS_EXAMPLE=5000
BUFFER_PER_CLASS=32

PER_TASK_EXAMPLE=$(($PER_CLASS_EXAMPLE * $PER_TASK_CLASS))
echo $PER_TASK_EXAMPLE
