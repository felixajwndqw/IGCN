python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 64 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx --single
python .\classify.py --dataset mnistrot --kernel_size 5 --base_channels 64 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx --single
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.35 --pooling maxmag --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.4 --pooling maxmag --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.45 --pooling maxmag --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.5 --pooling maxmag --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.55 --pooling avg --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.6 --pooling avg --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.55 --pooling maxmag --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --dropout 0.6 --pooling maxmag --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx


REM DONE
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 16 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 64 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 64 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 64 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 64 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 2 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 4 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 128 --no_g 16 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 3 --base_channels 96 --no_g 32 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 3 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 5 --base_channels 32 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 7 --base_channels 32 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 9 --base_channels 32 --no_g 8 --epochs 100 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 7 --base_channels 64 --no_g 8 --epochs 300 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
REM python .\classify.py --dataset mnistrot --kernel_size 7 --base_channels 128 --no_g 8 --epochs 300 --lr 1e-4 --weight_decay 1e-7 --splits 5 --inter_mg --final_mg --cmplx
