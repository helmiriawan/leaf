#!/usr/bin/env bash

NAME="fashion-mnist"

cd ../utils

python3 stats_keras.py --name $NAME

cd ../$NAME