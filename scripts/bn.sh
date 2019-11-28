#!/bin/bash

python ../main.py --name=bn_exp --train=True --SGD=False --bn=True  --device=cuda:0 --batch_size=512
