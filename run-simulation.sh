#!/bin/sh

#PBS -N abm-migration-trust
#PBS -l select=1:ncpus=10:mem=16gb
#PBS -l walltime=2:00:00

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate migration-trust
cd $HOME/abm_trust

#python simulation.py @args.txt
python simulation.py ${FILE}
