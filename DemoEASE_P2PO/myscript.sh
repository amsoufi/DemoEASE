#!/bin/bash

now=`date +%m_%d_%H:%M`

for param in 5 2 1
do
  for repeat in 0 1 2 3 4
  do
    free -m
    ../venv/bin/python3.8 run.py $now $repeat $param
    free -m
  done
done
