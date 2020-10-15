CUDA_VISIBLE_DEVICES=2,3 python train.py \
    -t /mnt/hdd/yangwh/benchmark_dataset/bsd_label/ \
    -tc  /mnt/hdd/yangwh/benchmark_dataset/bsd_22/ \
    -t1 /mnt/hdd/yangwh/benchmark_dataset/label/ \
    -tc1 /mnt/hdd/yangwh/benchmark_dataset/22/ \
    -v /mnt/hdd/yangwh/benchmark_dataset/val_label/ \
    -vc /mnt/hdd/yangwh/benchmark_dataset/val_22/ \
    -q 22 \
    -b 30 \
    -bs 28 \
    -md /mnt/hdd/yangwh/compression-artifacts-becnmark-models/models/PRN/QP22/ \
    -l 0.0001 \
    -e 60

