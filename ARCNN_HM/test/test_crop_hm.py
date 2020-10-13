import os
import sys
import time
import torch
import argparse
from torch import nn, optim
import torchvision as tv

sys.path.append("..")

import utils_test_hm as utils_test
import model_arcnn
from common import *

def ensure_exists(dname):
    import os
    if not os.path.exists(dname):
        try:
            os.makedirs(dname)
        except:
            pass
    return dname

def save_model(model, optimizer, model_filename):
    state_dict = {
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }


    print("Save %s" %model_filename)
    torch.save(state_dict, model_filename)

def load_model(model, optimizer, model_filename, epoch):
    print("Load %s" %model_filename)
    state_dict = torch.load(model_filename)
    
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    model.epoch = epoch ## reset model epoch

    return model, optimizer

def testNet(net, val_loader, val_set, out_dir, crop):
    fout = open(os.path.join(out_dir, 'train.log'), 'a')

    net.eval()

    timestamp = time.time()
    with torch.no_grad():
         print(crop)
         print(type(crop))
         p_psnr = utils_test.evalPsnr(net, val_loader, val_set, fout=fout, out_dir=out_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--val_dir', '-v', type=str,  default='augv/')
parser.add_argument('--val_cdir', '-vc', type=str,  default='augv_QF10/')
parser.add_argument('--qf', '-q', type=int, default=10)
parser.add_argument('--modeldir', '-md', type=str, default='res/')
parser.add_argument('--outputdir', '-od', type=str, default='res/')
parser.add_argument('--crop', '-cr', type=bool, default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    qf_fullname = 'matlab' + str(args.qf)

    val_set = utils_test.ImageDir(args.val_dir, args.val_cdir, preload=False,
                             transform=tv.transforms.Compose([
                                 utils_test.Align2(8),
                                 utils_test.GenerateMultiscale(),
                                 utils_test.ToTorchTensor()
                             ]), is_train=False)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=1)
    ensure_exists(args.outputdir)

    jnet = model_arcnn.ARCNN()
    print(sum(param.numel() for param in jnet.parameters()))

    jnet = nn.DataParallel(jnet).cuda()
    jnet.load_state_dict(torch.load(args.modeldir))

    testNet(jnet, val_loader, val_set, args.outputdir, True)
