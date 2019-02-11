#!/usr/bin/env bash

NAME="mnist"

cd ../utils

python3 stats_mnist.py --name $NAME

cd ../$NAME