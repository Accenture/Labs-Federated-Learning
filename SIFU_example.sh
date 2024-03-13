# This script can be used to reproduce the performances of SIFU on CIFAR10 for Figure 2 of our paper.

# First one run without parallelization in case the data is not downloaded at launch
python3 FU.py --dataset_name CIFAR10_0.1 --forgetting P0 --unlearn_scheme train --T 10000 --n_SGD 5 --B 20 --lr_l 0.01 --M 100 --n_sampled 5 --lambd 0.0 --stop_acc 90 --epsilon 10 --sigma 0.05 --seed 0 --compute_diff True --clip 10000 --model CNN

# Then all training runs
for i in {1..9}
do
    python3 FU.py --dataset_name CIFAR10_0.1 --forgetting P0 --unlearn_scheme train --T 10000 --n_SGD 5 --B 20 --lr_l 0.01 --M 100 --n_sampled 5 --lambd 0.0 --stop_acc 90 --epsilon 10 --sigma 0.05 --seed $i --compute_diff True --clip 10000 --model CNN &
done
wait

policy="P70"
# And the unlearning, with one loop per value of the limit_train_iter argument
for i in {0..9}
do
        python3 FU.py --dataset_name CIFAR10_0.1 --forgetting $policy --unlearn_scheme $scheme --T 10000 --n_SGD 5 --B 20 --lr_l 0.01 --M 100 --n_sampled 5 --lambd 0.0 --stop_acc 90 --epsilon 10 --sigma 0.05 --seed $i --compute_diff True --clip 10000 --model CNN --limit_train_iter 0.25 &
done
wait

for i in {0..9}
do
    python3 FU.py --dataset_name CIFAR10_0.1 --forgetting $policy --unlearn_scheme $scheme --T 10000 --n_SGD 5 --B 20 --lr_l 0.01 --M 100 --n_sampled 5 --lambd 0.0 --stop_acc 90 --epsilon 10 --sigma 0.05 --seed $i --compute_diff True --clip 10000 --model CNN --limit_train_iter 0.333 &
done
wait


for i in {0..9}
do
    python3 FU.py --dataset_name CIFAR10_0.1 --forgetting $policy --unlearn_scheme $scheme --T 10000 --n_SGD 5 --B 20 --lr_l 0.01 --M 100 --n_sampled 5 --lambd 0.0 --stop_acc 90 --epsilon 10 --sigma 0.05 --seed $i --compute_diff True --clip 10000 --model CNN --limit_train_iter 0.5 &
done
wait

