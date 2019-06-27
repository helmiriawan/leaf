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
    --nu)
    IUSER="$2"
    shift # past argument
    if [ ${IUSER:0:1} = "-" ]; then
        IUSER=""
    else
        shift # past value
    fi
    ;;
    --ns)
    TRAIN="$2"
    shift # past argument
    if [ ${TRAIN:0:1} = "-" ]; then
        TRAIN=""
    else
        shift # past value
    fi
    ;;
    --nt)
    TEST="$2"
    shift # past argument
    if [ ${TEST:0:1} = "-" ]; then
        TEST=""
    else
        shift # past value
    fi
    ;;
    --np)
    PARTITIONS="$2"
    shift # past argument
    if [ ${PARTITIONS:0:1} = "-" ]; then
        PARTITIONS=""
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

NUMBER_USER=""
if [ ! -z $IUSER ]; then
    NUMBER_USER="--nu $IUSER"
fi
NUMBER_TRAIN=""
if [ ! -z $TRAIN ]; then
    NUMBER_TRAIN="--ns $TRAIN"
fi
NUMBER_TEST=""
if [ ! -z $TEST ]; then
    NUMBER_TEST="--nt $TEST"
fi
NUMBER_PARTITIONS=""
if [ ! -z $PARTITIONS ]; then
    NUMBER_PARTITIONS="--np $PARTITIONS"
fi

if [ ! $SAMPLE = "na" ]; then
    if [ $SAMPLE = "iid" ]; then
        python3 mnist.py --name $NAME --iid $NUMBER_USER $NUMBER_TRAIN $NUMBER_TEST $NUMBER_PARTITIONS
    else
        python3 mnist.py --name $NAME $NUMBER_USER $NUMBER_TRAIN $NUMBER_TEST $NUMBER_PARTITIONS
    fi
fi

cd ../$NAME
