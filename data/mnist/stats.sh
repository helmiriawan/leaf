#!/usr/bin/env bash

NAME="mnist"

cd ../utils

python3 stats_keras.py --name $NAME

cd ../$NAME