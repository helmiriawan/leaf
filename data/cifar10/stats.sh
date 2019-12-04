#!/usr/bin/env bash

NAME="cifar10"

cd ../utils

python3 stats_keras.py --name $NAME

cd ../$NAME
