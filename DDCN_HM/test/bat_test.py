import os

dataset_str = ['LIVE1', 'classic5', 'BSD100', 'Liu4k'];
postfix_str = ['300', '300', '300', '300']
QP_list = [22, 27, 32, 37]

for ds in range(0, 2):
    for qf in range(3, 4):
        cmd = '''QP='''+str(QP_list[qf])+'''
    CUDA_VISIBLE_DEVICES=5 python test_crop_hm.py \
        -v /mnt/ssd/benchmark_dataset/TEST_SETS/'''+dataset_str[ds]+'''_label_pad/ \
        -vc /mnt/ssd/benchmark_dataset/TEST_SETS/'''+dataset_str[ds]+'''_QP$QP/ \
        -q $QP \
        -md  /mnt/ssd/yangwh/compression-artifacts-becnmark/models/DDCN/QP$QP/ckpt/''' + postfix_str[qf]+''' \
        -od /mnt/hdd/yangwh/compression-artifacts-becnmark/outputs/DDCN/QP$QP/'''+dataset_str[ds]

        if ds<=2:
            cmd+= '''/ \
        -cr False'''
        else:
            cmd+= '''/ \
        -cr True'''

        print(cmd)
        os.system(cmd)
