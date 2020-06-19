#!/bin/sh
cat txt_to_run_experiments/simple_experiments.txt |while read line;do
python ./one_freerider.py $line
done
