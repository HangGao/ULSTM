# script for experiments in parallel

CUDA_VISIBLE_DEVICES=0 python review.py --model PLSTM --nlayers 1 --nhid 50 --save save/PLSTM_1_50.pt > save/PLSTM_1_50.out &
CUDA_VISIBLE_DEVICES=0 python review.py --model PLSTM --nlayers 1 --nhid 100 --save save/PLSTM_1_100.pt > save/PLSTM_1_100.out &
CUDA_VISIBLE_DEVICES=1 python review.py --model PLSTM --nlayers 1 --nhid 150 --save save/PLSTM_1_150.pt > save/PLSTM_1_150.out &
CUDA_VISIBLE_DEVICES=1 python review.py --model PLSTM --nlayers 1 --nhid 200 --save save/PLSTM_1_200.pt > save/PLSTM_1_200.out &
CUDA_VISIBLE_DEVICES=2 python review.py --model PLSTM --nlayers 1 --nhid 250 --save save/PLSTM_1_250.pt > save/PLSTM_1_250.out &
CUDA_VISIBLE_DEVICES=2 python review.py --model PLSTM --nlayers 1 --nhid 300 --save save/PLSTM_1_300.pt > save/PLSTM_1_300.out & 
