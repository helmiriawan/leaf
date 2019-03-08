#!/usr/bin/env bash

# download data and convert to .json format

# --------------------
# parse arguments

IUSER="" # --iu tag, # of users
SAMPLES="na" # -k tag, # of samples per user
TARGET="" # -y tag, sensor or actuator in the system

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
    --iu)
    IUSER="$2"
    shift # past argument
    if [ ${IUSER:0:1} = "-" ]; then
        IUSER=""
    else
        shift # past value
    fi
    ;;
    -k)
    SAMPLES="$2"
    shift # past argument
    if [ ${SAMPLES:0:1} = "-" ]; then
        SAMPLES=""
    else
        shift # past value
    fi
    ;;
    -y)
    TARGET="$2"
    shift # past argument
    if [ ${TARGET:0:1} = "-" ]; then
        TARGET=""
    else
        shift # past value
    fi
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done

NAME="swat" # name of the dataset, equivalent to directory name

cd ../utils

# sample data
IUSERTAG=""
if [ ! -z $IUSER ]; then
    IUSERTAG="--u $IUSER"
fi
if [ ! -z $TARGET ]; then
    TARGET_TAG="--target $TARGET"
fi

python3 swat.py --name $NAME $IUSERTAG $TARGET_TAG --samples $SAMPLES

cd ../$NAME
