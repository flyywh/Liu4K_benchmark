## A Comprehensive Benchmark for Single Image Compression Artifacts Reduction (CVPR'2018)

[Jiaying Liu](http://39.96.165.147/people/liujiaying.html), 
[Dong Liu](http://staff.ustc.edu.cn/~dongeliu/), 
[Wenhan Yang](https://flyywh.github.io/), 
[Sifeng Xia](http://39.96.165.147/people/xsf.html),
[Xiaoshuai Zhang](https://i.buriedjet.com/),
and Yuanying Dai

[[Paper Link]](https://arxiv.org/abs/1909.03647)

### Abstract

We present a comprehensive study and evaluation of existing single image compression artifacts removal algorithms, using a new 4K resolution benchmark including diversified foreground objects and background scenes with rich structures, called Large-scale Ideal Ultra high definition 4K (LIU4K) benchmark. Compression artifacts removal, as a common post-processing technique, aims at alleviating undesirable artifacts such as blockiness, ringing, and banding caused by quantization and approximation in the compression process. In this work, a systematic listing of the reviewed methods is presented based on their basic models (handcrafted models and deep networks). The main contributions and novelties of these methods are highlighted, and the main development directions, including architectures, multi-domain sources, signal structures, and new targeted units, are summarized. Furthermore, based on a unified deep learning configuration (i.e. same training data, loss function, optimization algorithm, etc.), we evaluate recent deep learning-based methods based on diversified evaluation measures. The experimental results show the state-of-the-art performance comparison of existing methods based on both full-reference, non-reference and task-driven metrics. Our survey would give a comprehensive reference source for future research on single image compression artifacts removal and inspire new directions of the related fields.

#### If you find the resource useful, please cite the following :- )

```
@ARTICLE{LIU4K,
author={J. {Liu} and D. {Liu} and W. {Yang} and S. {Xia} and X. {Zhang} and Y. {Dai}},
journal={IEEE Transactions on Image Processing}, 
title={A Comprehensive Benchmark for Single Image Compression Artifact Reduction}, 
year={2020},
volume={29},
number={},
pages={7845-7860},
}
```

## Dataset

The whole dataset can be found on Baidu:

https://pan.baidu.com/s/1ND5nYP2CJ1g1vW8KpzsCXw (extracted code: w9f0)

or on Google Drive:

https://drive.google.com/drive/folders/1DqN4IAkWCwcCDoaHVmZJqp-6YwzkC_9k (Training Set)

https://drive.google.com/drive/folders/1Q5X8UTAKnodfwiQq2tf7ynj6MdTlq3dF (Validation Set)

The images in the dataset are under the license CC BY-NC-ND 4.0. 

Thanks to Yueyu Hu and Datong Wei for providing part of the images in the dataset (see list.txt).

## Testing Example
You can refer to ./DnCNN_HM/test/test.sh for testing command usage.

## Training Example
You can refer to ./DnCNN_HM/train_DnCNN_HM_QP22.sh for training command usage.

## Contact

If you have questions, you can contact `yangwenhan@pku.edu.cn`.


