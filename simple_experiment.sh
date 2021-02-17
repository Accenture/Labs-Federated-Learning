#!/bin/sh
cat txt_to_run_experiments/simple_experiments.txt |while read line;do
python ./free-riding.py $line
done
python ./python_code/simple_plot.py
