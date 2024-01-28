#!/bin/bash
docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/FA-ExU-Net/ -v /your/exp_result/path:/results -v /your/data/path:/data docker

