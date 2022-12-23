#!/bin/bash

if [ $1 == "train" ]; then
    #python train.py --dataset cats_vs_dogs --data_path ./_data/888_v0/train --epochs 100 --eval_every 1
    python train.py --dataset cats_vs_dogs --data_path ./_data/mam_v1/train --epochs 100 --eval_every 1
elif [ $1 == "eval" ]; then
    #python eval.py --dataset cats_vs_dogs --data_path ./_data/888_v0/eval
    #python eval.py --dataset cats_vs_dogs --data_path ./_data/mam_v0/eval
    python eval.py --dataset cats_vs_dogs --data_path ./_data/mam_v0_unlabeled/test
fi