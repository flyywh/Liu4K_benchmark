import os
import time
import torch
import argparse
from torch import nn, optim
import torchvision as tv
import numpy as np
import utils
import model_dmcnn

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    print('occupy success!')

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

    model.epoch = epoch

    return model, optimizer

def trainNet(net, opt, cri, sch, cp_dir, epoch_range, training_set, val_set,batch_size_big, rep=1):
    tb_dir = os.path.join(cp_dir, 'tb')
    utils.ensure_exists(tb_dir)
    fout = open(os.path.join(cp_dir, 'train.log'), 'a')

    for epoch in epoch_range:
        net.train()
        running_loss = 0
        start_time = time.time()
        tl = len(training_set)

        for e in range(rep):
            for i, data in enumerate(training_set, 0):

                com = data['com'].float().cuda()
                c_2 = data['com_2'].float().cuda()
                c_4 = data['com_4'].float().cuda()
                org = data['org'].float().cuda()
                o_2 = data['org_2'].float().cuda()
                o_4 = data['org_4'].float().cuda()               

                com_pair = (c_4, c_2, com)
                org_pair = (o_4, o_2, org)

                opt.zero_grad()
                ret = net(com_pair)
                loss, MSE4, MSE2, MSEp, MSEd = cri(ret, org_pair)

                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10)
                opt.step()

                running_loss += loss.item()
                if i%100==0:
                    print('[Running epoch %2d, batch %4d] loss: %.3f' %
                          (epoch + 1, i + 1, \
                           10000 * running_loss / (e * tl + i + 1),
                           ), end='\n')
                else:                
                    print('[Running epoch %2d, batch %4d] loss: %.3f' %
                          (epoch + 1, i + 1, \
                           10000 * running_loss / (e * tl + i + 1),
                           ), end='\r')
                
        if not (epoch + 1) % 1:
            timestamp = time.time()
            print('[timestamp %d, epoch %2d] loss: %.3f, time: %6ds        ' %
                  (timestamp, epoch + 1, 10000 * running_loss / ((i + 1) * rep),
                   timestamp - start_time), end='\n')
            with torch.no_grad():
                p_psnr = utils.evalPsnr(net, val_set, fout=fout)

            save_model(net,opt,os.path.join(cp_dir, str(epoch + 1)+'_withopt'))
            torch.save(net.state_dict(), os.path.join(cp_dir, str(epoch + 1) ))
            sch.step()
            print('cur_lr: %.5f' % sch.get_lr()[0])


parser = argparse.ArgumentParser()
parser.add_argument('--training-dir', '-t', type=str,  default=None)
parser.add_argument('--training-cdir', '-tc', type=str,  default=None)
parser.add_argument('--training-dir1', '-t1', type=str,  default=None)
parser.add_argument('--training-cdir1', '-tc1', type=str,  default=None)
parser.add_argument('--training-dir2', '-t2', type=str,  default=None)
parser.add_argument('--training-cdir2', '-tc2', type=str,  default=None)
parser.add_argument('--val-dir', '-v', type=str,  default='augv/')
parser.add_argument('--val-cdir', '-vc', type=str,  default='augv_QF10/')
parser.add_argument('--qf', '-q', type=int, default=10)
parser.add_argument('--batch-size', '-b', type=int, default=1)
parser.add_argument('--learning-rate', '-l', type=float, default=0.001)
parser.add_argument('--epoch_num', '-e', type=int, default=80)
parser.add_argument('--modeldir', '-md', type=str, default='res/')
parser.add_argument('--batch-size-small', '-bs', type=int, default=1)


if __name__ == "__main__":
    args = parser.parse_args()
    qf_fullname = 'matlab' + str(args.qf)


    training_set = utils.ImageDir(args.training_dir, args.training_cdir, args.training_dir1, args.training_cdir1, args.training_dir2, args.training_cdir2,preload=False,
                                  transform=tv.transforms.Compose([
                                      utils.RandomCrop(56),
                                      utils.GenerateMultiscale(),
                                      utils.ToTorchTensor()
                                  ]))

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=args.batch_size, num_workers=1, shuffle=True)

    val_set = utils.ImageDir(args.val_dir, args.val_cdir, preload=False,
                             transform=tv.transforms.Compose([
                                 utils.ValCrop(448),
                                 utils.Align2(8),
                                 utils.GenerateMultiscale(),
                                 utils.ToTorchTensor()
                             ]))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=1)


    epoch_mul=5
    
    jnet = nn.DataParallel(model_dmcnn.DMCNN(1)).cuda()

    if args.qf > 22:
        jnet.load_state_dict(torch.load('/mnt/ssd/yangwh/compression-artifacts-becnmark/models/DDCN/QP' + str(args.qf-5) + '/ckpt/' + str(epoch_mul*args.epoch_num)))
#    cri = torch.nn.MSELoss().cuda()
    cri = model_dmcnn.dmcnnLoss().cuda()

    opt = optim.Adam(filter(lambda p: p.requires_grad, jnet.parameters()), lr=args.learning_rate)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 16, eta_min=args.learning_rate)

    trainNet(jnet, opt, cri, sch, args.modeldir + './ckpt', range(7*epoch_mul), training_loader,val_loader,args.batch_size-args.batch_size_small, rep=1)

    training_set.transform = tv.transforms.Compose([
        utils.RandomCrop(112),
        utils.GenerateMultiscale(),
        utils.ToTorchTensor()
    ])

    trainNet(jnet, opt, cri, sch, args.modeldir + './ckpt', range(7*epoch_mul, 10*epoch_mul), training_loader, val_loader,args.batch_size-args.batch_size_small, rep=1)

    training_set.transform = tv.transforms.Compose([
        utils.RandomCrop(168),
        utils.GenerateMultiscale(),
        utils.ToTorchTensor()
    ])
    trainNet(jnet, opt, cri, sch, args.modeldir + './ckpt', range(10*epoch_mul, 13*epoch_mul), training_loader, val_loader,args.batch_size-args.batch_size_small, rep=1)

    training_set.transform = tv.transforms.Compose([
        utils.RandomCrop(224),
        utils.GenerateMultiscale(),
        utils.ToTorchTensor()
    ])

    trainNet(jnet, opt, cri, sch, args.modeldir + './ckpt', range(13*epoch_mul, 16*epoch_mul), training_loader, val_loader,args.batch_size-args.batch_size_small, rep=1)

    training_set.transform = tv.transforms.Compose([
        utils.RandomCrop(256),
        utils.GenerateMultiscale(),
        utils.ToTorchTensor()
    ])

    opt = optim.SGD(filter(lambda p: p.requires_grad, jnet.parameters()), lr=args.learning_rate )#/ 10
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 16)

    trainNet(jnet, opt, cri, sch, args.modeldir + './ckpt', range(16*epoch_mul, args.epoch_num*epoch_mul), training_loader, val_loader,args.batch_size-args.batch_size_small,rep=1)
