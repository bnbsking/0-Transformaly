#!/bin/bash

if [ $1 == "train" ]; then
    python train.py --dataset cats_vs_dogs --data_path ./_data/mam/train --epochs 100 --eval_every 1
elif [ $1 == "eval" ]; then
    python eval.py --dataset cats_vs_dogs --data_path ./_data/mam/eval
fi