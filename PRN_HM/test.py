import torch
import torchvision as tv
from torch import nn
import argparse

import utils
import prn_b_10 as model_prn
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--testing-dir', '-t', type=str,  default='/mnt/ssd/wangdezhao/val_label')
parser.add_argument('--testing-cdir', '-tc', type=str,  default='/mnt/ssd/wangdezhao/val_37')
parser.add_argument('--checkpoint', '-c', type=str, default='/mnt/hdd/compression-artifacts-becnmark/models/HRD4K/PRN/QP37/ckpt/300')
parser.add_argument('--qf', '-q', type=int, default=10)
# parser.add_argument('--output-dir', '-o', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    qf_fullname = 'matlab' + str(args.qf)

    testing_set = utils.ImageDir(args.testing_dir, args.testing_cdir, preload=False,
                             transform=tv.transforms.Compose([
                                     
                                     # utils.ValCrop(448),
                                     utils.Align2(8),
                                     # utils.GenerateJPEGPair(args.training_dir,args.qf),
                                     
                                     utils.GenerateMultiscale(),
                                     utils.ToTorchTensor()
                                 ]))
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, num_workers=1)

    # jnet = nn.DataParallel(model_prn.PRN(),device_ids=[1]).cuda()
    torch.cuda.set_device(1)
    jnet = model_prn.PRN().cuda()
    w = torch.load(args.checkpoint)
    new_w = collections.OrderedDict()
    for key,v in w.items():
        new_w[key[7:]] = w[key]
    # jnet.load_state_dict(torch.load(args.checkpoint))
    jnet.load_state_dict(new_w)

    with torch.no_grad():
        # utils.evalPsnr(jnet, testing_loader)
        utils.evalPsnrPartition(jnet,testing_loader)
        # if args.output_dir is not None:
        #     utils.testAndSave(jnet, testing_loader, args.output_dir)
