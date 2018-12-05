CUDA_VISIBLE_DEVICES=0 python main.py --method='CMOT' --ds_no=3
CUDA_VISIBLE_DEVICES=0 python main.py --method='CMOT' --ds_no=3 --use_ssm

CUDA_VISIBLE_DEVICES=0 python main.py --method='SCEA' --ds_no=3
CUDA_VISIBLE_DEVICES=1 python main.py --method='SCEA' --ds_no=3 --use_ssm

CUDA_VISIBLE_DEVICES=1 python main.py --method='ELP' --ds_no=3
CUDA_VISIBLE_DEVICES=1 python main.py --method='ELP' --ds_no=3 --use_ssm
