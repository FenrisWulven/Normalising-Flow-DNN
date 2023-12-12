#!/bin/sh
#BSUB -q gpuv100                # specify queue
#BSUB -J test                  # job name
#BSUB -o test_%J.out            # output name
#BSUB -e test_%J.err		 # output name (error)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R span[hosts=1]         # cores must be on the same host
#BSUB -R "rusage[mem=4GB]" # kun cpu?
#BSUB -n 8                       # number of cores
#BSUB -W 6:00                       # set walltime limit hhmm
#BSUB -u s204070@dtu.dk          # email
#BSUB -N                       # notify when job end

###batch size, memory, cores

#module unload cuda
#module unload cudnn
module load python3/3.10.7
module load cuda/12.1.1
module load cudnn/v8.9.1.23-prod-cuda-12.X
#module load cudnn/v8.3.2.44-prod-cuda-11.X

#module load cuda/12.1.1
#module load cudnn/v8.6.0.163-prod-cuda-11.X

###/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
source venv3/bin/activate
python3 postnet/run.py
#python3 ensemble/run_ensemble.py

