CUDA_VISIBLE_DEVICES=3,6 python new_train.py \
    -t /mnt/ssd/benchmark_dataset/bsd_label/ \
    -tc  /mnt/ssd/benchmark_dataset/bsd_32/ \
    -t1 /mnt/ssd/benchmark_dataset/label/ \
    -tc1 /mnt/ssd/benchmark_dataset/32/ \
    -v /mnt/ssd/benchmark_dataset/val_label/ \
    -vc /mnt/ssd/benchmark_dataset/val_32/ \
    -q 32 \
    -b 30 \
    -bs 28 \
    -md /mnt/ssd/yangwh/compression-artifacts-becnmark/models/DDCN/QP32/ \
    -l 0.0001 \
    -e 60
 
