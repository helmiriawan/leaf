#!/usr/bin/env bash

# download data and convert to .json format

# --------------------
# parse arguments

SAMPLE="na" # -s tag, iid or niid

if [ ! -d "data" ]; then
  mkdir data
fi

if [ ! -d "data/train" ]; then
  mkdir data/train
fi

if [ ! -d "data/test" ]; then
  mkdir data/test
fi

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s)
    SAMPLE="$2"
    shift # past argument
    if [ ${SAMPLE:0:1} = "-" ]; then
        SAMPLE=""
    else
        shift # past value
    fi
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

NAME="mnist" # name of the dataset, equivalent to directory name

cd ../utils

if [ ! $SAMPLE = "na" ]; then
    if [ $SAMPLE = "iid" ]; then
        python3 mnist.py --name $NAME --iid
    else
        python3 mnist.py --name $NAME
    fi
fi

cd ../$NAME
