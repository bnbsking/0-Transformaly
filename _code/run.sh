#!/bin/bash

if [ $1 == "d1s1" ]; then
    python train.py --dataset cats_vs_dogs --data_path ./_data/example/train --batch_size 4 --dataset_index d1 --epochs 10 --eval_every 1 
elif [ $1 == "d1s2" ]; then
    python eval.py  --dataset cats_vs_dogs --data_path ./_data/example/train --batch_size 4 --dataset_index d1
elif [ $1 == "d2s1" ]; then
    python train.py --dataset cats_vs_dogs --data_path ./_data/example/test  --batch_size 4 --dataset_index d2
elif [ $1 == "d2s2" ]; then
    python eval.py  --dataset cats_vs_dogs --data_path ./_data/example/test  --batch_size 4 --dataset_index d2
fi
