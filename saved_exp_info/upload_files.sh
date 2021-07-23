

rsync -h -v -P -t nef-frontal.inria.fr:sampling_review/saved_exp_info/acc/* ./acc/

rsync -h -v -P -t nef-frontal.inria.fr:sampling_review/saved_exp_info/loss/* ./loss/

rsync -h -v -P -t nef-frontal.inria.fr:sampling_review/saved_exp_info/sampled_clients/* ./sampled_clients/

rsync -h -v -P -t nef-frontal.inria.fr:sampling_review/saved_exp_info/len_dbs/* ./len_dbs/

rsync -h -v -P -t nef-frontal.inria.fr:sampling_review/saved_exp_info/agg_weights/* ./agg_weights/

cd ..


printf "\n ---> Remaining experiments"
python txt_experiments.py


printf "\n ---> Formating Properly the .py files to meet the Pep8 guidelines\n"
autopep8 --in-place --recursive . &> /dev/null
