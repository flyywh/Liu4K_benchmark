CUDA_VISIBLE_DEVICES=3,6 python train.py \
    -t /mnt/ssd/benchmark_dataset/bsd_label/ \
    -tc  /mnt/ssd/benchmark_dataset/bsd_27/ \
    -t1 /mnt/ssd/benchmark_dataset/label/ \
    -tc1 /mnt/ssd/benchmark_dataset/27/ \
    -v /mnt/ssd/benchmark_dataset/val_label/ \
    -vc /mnt/ssd/benchmark_dataset/val_27/ \
    -q 27 \
    -b 30 \
    -bs 28 \
    -md /mnt/ssd/yangwh/compression-artifacts-becnmark/models/DDCN/QP27/ \
    -l 0.0001 \
    -e 60
 
