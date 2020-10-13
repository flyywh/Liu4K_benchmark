CUDA_VISIBLE_DEVICES=3,6 python new_train.py \
    -t /mnt/ssd/benchmark_dataset/bsd_label/ \
    -tc  /mnt/ssd/benchmark_dataset/bsd_37/ \
    -t1 /mnt/ssd/benchmark_dataset/label/ \
    -tc1 /mnt/ssd/benchmark_dataset/37/ \
    -v /mnt/ssd/benchmark_dataset/val_label/ \
    -vc /mnt/ssd/benchmark_dataset/val_37/ \
    -q 37 \
    -b 30 \
    -bs 28 \
    -md /mnt/ssd/yangwh/compression-artifacts-becnmark/models/DDCN/QP37/ \
    -l 0.0001 \
    -e 60
 
