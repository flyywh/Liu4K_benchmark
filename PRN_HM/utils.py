import os
import torch
import numpy as np
from io import BytesIO
from PIL import Image as im
from PIL import JpegPresets
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
import torchvision.transforms as transforms
import copy
# use same QT with matlab
JpegPresets.presets['matlab20'] = {
    'quantization': [
        [
            40,28,25,40,60,100,128,153,
            30,30,35,48,65,145,150,138,
            35,33,40,60,100,143,173,140,
            35,43,55,73,128,218,200,155,
            45,55,93,140,170,255,255,193,
            60,88,138,160,203,255,255,230,
            123,160,195,218,255,255,255,253,
            180,230,238,245,255,250,255,248
        ], [
            40,28,25,40,60,100,128,153,
            30,30,35,48,65,145,150,138,
            35,33,40,60,100,143,173,140,
            35,43,55,73,128,218,200,155,
            45,55,93,140,170,255,255,193,
            60,88,138,160,203,255,255,230,
            123,160,195,218,255,255,255,253,
            180,230,238,245,255,250,255,248
        ]],
    'subsampling': 0
}

JpegPresets.presets['matlab10'] = {
    'quantization': [
        [
            80,55,50,80,120,200,255,255,
            60,60,70,95,130,255,255,255,
            70,65,80,120,200,255,255,255,
            70,85,110,145,255,255,255,255,
            90,110,185,255,255,255,255,255,
            120,175,255,255,255,255,255,255,
            245,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255
        ], [
            80,55,50,80,120,200,255,255,
            60,60,70,95,130,255,255,255,
            70,65,80,120,200,255,255,255,
            70,85,110,145,255,255,255,255,
            90,110,185,255,255,255,255,255,
            120,175,255,255,255,255,255,255,
            245,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255
        ]],
    'subsampling': 0
}


def ensure_exists(dname):
    import os
    if not os.path.exists(dname):
        try:
            os.makedirs(dname)
        except:
            pass
    return dname

class ImageDirColor(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, root_cdir=None, root_dir1=None, root_cdir1=None,root_dir2=None, root_cdir2=None, transform=None, preload=False):
        self.root_dir = root_dir
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform
        self.image_list = []
        self.image_clist = []

        if root_dir is not None:
            for r, d, filenames in os.walk(self.root_dir):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png']:# or '=1' in f or '=2' in f
                        continue

                    self.image_list.append(os.path.join(r, f))
                    self.image_clist.append(os.path.join(root_cdir, f[:-4]+'.png'))

        if root_dir1 is not None:
            for r, d, filenames in os.walk(root_cdir1):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png'] or '=1' in f or '=2' in f or 'cuave' in f:# 
                        continue
                    self.image_clist.append(os.path.join(r, f))
                    self.image_list.append(os.path.join(root_dir1, f[:-9]+'label.png'))
        if root_dir2 is not None:
            for r, d, filenames in os.walk(self.root_dir2):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png'] or '=1' in f or '=2' in f:# 
                        continue
                    self.image_list.append(os.path.join(r, f))
                    self.image_clist.append(os.path.join(root_cdir2, f[:-9]+'block.png'))

        self.loaded_images = [None] * len(self.image_list)
        self.loaded_cimages = [None] * len(self.image_list)

        if preload:
            for idx in range(len(self.image_list)):
                tmp = plt.imread(self.image_list[idx])
                if tmp.dtype == np.float32:
                    tmp = np.asarray(tmp * 255, dtype=np.uint8)
                self.loaded_images[idx] = tmp

                tmp = plt.imread(self.image_clist[idx])
                if tmp.dtype == np.float32:
                    tmp = np.asarray(tmp * 255, dtype=np.uint8)
                self.loaded_cimages[idx] = tmp

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.loaded_images[idx] is None:
            tmp = im.open(self.image_list[idx])
            tmp = np.asarray(tmp)
            #if tmp.dtype == np.float32:
            #    tmp = np.asarray(tmp * 255, dtype=np.uint8)
            #self.loaded_images[idx] = tmp

            tmpc = im.open(self.image_clist[idx])
            tmpc = np.asarray(tmpc)
            #if tmpc.dtype == np.float32:
            #    tmpc = np.asarray(tmpc * 255, dtype=np.uint8)
            #self.loaded_cimages[idx] = tmp

        ret = {'org': tmp, 'com': tmpc}

        if self.transform:
            ret = self.transform(ret)

        ret['org_path'] = self.image_list[idx]
        ret['com_path'] = self.image_clist[idx]

        return ret


class ImageDir(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, root_cdir=None, root_dir1=None, root_cdir1=None,root_dir2=None, root_cdir2=None, transform=None, preload=False):
        self.root_dir = root_dir
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform
        self.image_list = []
        self.image_clist = []

        #ADDED by WDZ
        self.cuave0_list = []
        self.cuave1_list = []
        self.cuave2_list = []
        self.cuave3_list = []

        if root_dir is not None:
            for r, d, filenames in os.walk(self.root_dir):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png']:# or '=1' in f or '=2' in f
                        continue
                    self.image_list.append(os.path.join(r, f))
                    self.image_clist.append(os.path.join(root_cdir, f[:-9]+'block.png'))

                    self.cuave0_list.append(os.path.join(root_cdir, f[:-9]+'cuave0.png'))
                    self.cuave1_list.append(os.path.join(root_cdir, f[:-9]+'cuave1.png'))
                    self.cuave2_list.append(os.path.join(root_cdir, f[:-9]+'cuave2.png'))
                    self.cuave3_list.append(os.path.join(root_cdir, f[:-9]+'cuave3.png'))
        if root_dir1 is not None:
            for r, d, filenames in os.walk(root_cdir1):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png'] or '=1' in f or '=2' in f or 'cuave' in f:# 
                        continue
                    self.image_clist.append(os.path.join(r, f))
                    self.image_list.append(os.path.join(root_dir1, f[:-9]+'label.png'))

                    self.cuave0_list.append(os.path.join(r, f[:-9]+'cuave0.png'))
                    self.cuave1_list.append(os.path.join(r, f[:-9]+'cuave1.png'))
                    self.cuave2_list.append(os.path.join(r, f[:-9]+'cuave2.png'))
                    self.cuave3_list.append(os.path.join(r, f[:-9]+'cuave3.png'))


        if root_dir2 is not None:
            for r, d, filenames in os.walk(self.root_dir2):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png'] or '=1' in f or '=2' in f:# 
                        continue
                    self.image_list.append(os.path.join(r, f))
                    self.image_clist.append(os.path.join(root_cdir2, f[:-9]+'block.png'))

                    self.cuave0_list.append(os.path.join(root_cdir2, f[:-9]+'cuave0.png'))
                    self.cuave1_list.append(os.path.join(root_cdir2, f[:-9]+'cuave1.png'))
                    self.cuave2_list.append(os.path.join(root_cdir2, f[:-9]+'cuave2.png'))
                    self.cuave3_list.append(os.path.join(root_cdir2, f[:-9]+'cuave3.png'))
        
        self.loaded_images = [None] * len(self.image_list)
        self.loaded_cimages = [None] * len(self.image_list)

        self.loaded_cuave0 = [None]*len(self.cuave0_list)
        self.loaded_cuave1 = [None]*len(self.cuave1_list)
        self.loaded_cuave2 = [None]*len(self.cuave2_list)
        self.loaded_cuave3 = [None]*len(self.cuave3_list)

        if preload:
            for idx in range(len(self.image_list)):
                tmp = plt.imread(self.image_list[idx])
                if tmp.dtype == np.float32:
                    tmp = np.asarray(tmp * 255, dtype=np.uint8)
                self.loaded_images[idx] = tmp

                tmp = plt.imread(self.image_clist[idx])
                if tmp.dtype == np.float32:
                    tmp = np.asarray(tmp * 255, dtype=np.uint8)
                self.loaded_cimages[idx] = tmp

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.loaded_images[idx] is None:
            tmp = plt.imread(self.image_list[idx])
            if tmp.dtype == np.float32:
                tmp = np.asarray(tmp * 255, dtype=np.uint8)
            # self.loaded_images[idx] = tmp

            tmpc = plt.imread(self.image_clist[idx])
            if tmpc.dtype == np.float32:
                tmpc = np.asarray(tmpc * 255, dtype=np.uint8)
            # self.loaded_cimages[idx] = tmp

            tcuave0 = plt.imread(self.cuave0_list[idx])
            if tcuave0.dtype == np.float32:
                tcuave0 = np.asarray(tcuave0 * 255, dtype=np.uint8)
            # self.loaded_cimages[idx] = tmp

            tcuave1 = plt.imread(self.cuave1_list[idx])
            if tcuave1.dtype == np.float32:
                tcuave1 = np.asarray(tcuave1 * 255, dtype=np.uint8)
            # self.loaded_cimages[idx] = tmp

            tcuave2 = plt.imread(self.cuave2_list[idx])
            if tcuave2.dtype == np.float32:
                tcuave2 = np.asarray(tcuave2 * 255, dtype=np.uint8)
            # self.loaded_cimages[idx] = tmp

            tcuave3 = plt.imread(self.cuave3_list[idx])
            if tcuave3.dtype == np.float32:
                tcuave3 = np.asarray(tcuave3 * 255, dtype=np.uint8)
            # self.loaded_cimages[idx] = tmp


        ret = {'org': tmp, 'com': tmpc, 'cuave0':tcuave0, 'cuave1':tcuave1, 'cuave2':tcuave2, 'cuave3':tcuave3}

        if self.transform:
            ret = self.transform(ret)
        return ret

    # def __getitem__(self, idx):
    #     if self.loaded_images[idx] is None:
    #         tmp = plt.imread(self.image_list[idx])
    #         if tmp.dtype == np.float32:
    #             tmp = np.asarray(tmp * 255, dtype=np.uint8)
    #         self.loaded_images[idx] = tmp
    #
    #         tmp = plt.imread(self.image_clist[idx])
    #         if tmp.dtype == np.float32:
    #             tmp = np.asarray(tmp * 255, dtype=np.uint8)
    #         self.loaded_cimages[idx] = tmp
    #
    #     ret = {'org': self.loaded_images[idx][:], 'com': self.loaded_cimages[idx][:]}
    #
    #     if self.transform:
    #         ret = self.transform(ret)
    #     return ret


# Predefined transformations
class Align2(object):
    def __init__(self, l):
        self.l = l

    def __call__(self, pair):
        image = pair['org']
        com = pair['com']
        cuave0 = pair['cuave0']
        cuave1 = pair['cuave1']
        cuave2 = pair['cuave2']
        cuave3 = pair['cuave3']

        h, w = image.shape[:2]
        new_h, new_w = self.l * (h // self.l), self.l * (w // self.l)

        image = image[:new_h, :new_w]
        com = com[:new_h, :new_w]
        cuave0 = cuave0[:new_h, :new_w]
        cuave1 = cuave1[:new_h, :new_w]
        cuave2 = cuave2[:new_h, :new_w]
        cuave3 = cuave3[:new_h, :new_w]
        return {'org': image, 'com': com, 'cuave0':cuave0, 'cuave1':cuave1, 'cuave2':cuave2, 'cuave3':cuave3}

class ValCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, pair):
        image = pair['org']
        com = pair['com']
        cuave0 = pair['cuave0']
        cuave1 = pair['cuave1']
        cuave2 = pair['cuave2']
        cuave3 = pair['cuave3']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)

        top = (h-new_h)//2
        left = (w-new_w)//2
        
        image = image[top: top + new_h,
                left: left + new_w]
        com = com[top: top + new_h,
                left: left + new_w]
        cuave0 = cuave0[top: top + new_h,
                left: left + new_w]
        cuave1 = cuave1[top: top + new_h,
                left: left + new_w]
        cuave2 = cuave2[top: top + new_h,
                left: left + new_w]
        cuave3 = cuave3[top: top + new_h,
                left: left + new_w]
        return {'org': image, 'com': com, 'cuave0':cuave0, 'cuave1':cuave1, 'cuave2':cuave2, 'cuave3':cuave3}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, pair):
        image = pair['org']
        com = pair['com']
        cuave0 = pair['cuave0']
        cuave1 = pair['cuave1']
        cuave2 = pair['cuave2']
        cuave3 = pair['cuave3']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]
        com = com[top: top + new_h,
                left: left + new_w]
        cuave0 = cuave0[top: top + new_h,
                left: left + new_w]
        cuave1 = cuave1[top: top + new_h,
                left: left + new_w]
        cuave2 = cuave2[top: top + new_h,
                left: left + new_w]
        cuave3 = cuave3[top: top + new_h,
                left: left + new_w]
        return {'org': image, 'com': com, 'cuave0':cuave0, 'cuave1':cuave1, 'cuave2':cuave2, 'cuave3':cuave3}


CompressBuffer = BytesIO()


class GenerateJPEGPair(object):
    def __init__(self, quality):
        assert quality in ['matlab5', 'matlab10', 'matlab20']
        self.q = quality

    def __call__(self, image):
        assert (len(image.shape) == 3 and image.shape[2] == 3) or len(image.shape) == 2

        def Compress(q):
            tmp = im.fromarray(image)
            CompressBuffer.seek(0)
            tmp.save(CompressBuffer, "JPEG", quality=q)
            CompressBuffer.seek(0)
            tmp = np.asarray(im.open(CompressBuffer))
            return tmp

        if isinstance(self.q, str):
            com = Compress(self.q)
        elif isinstance(self.q, list):
            raise NotImplementedError('Not Implemented For Multiple Qs')
        #             com = [Compress(q) for q in self.q]

        return {'org': image, 'com': com}


class GenerateMultiscale(object):
    def __init__(self):
        pass

    def __call__(self, pair):
        # print(pair)
        org = pair['org']
        com = pair['com']
        cuave0 = pair['cuave0']
        cuave1 = pair['cuave1']
        cuave2 = pair['cuave2']
        cuave3 = pair['cuave3']

        if len(org.shape) > 2:
            raise NotImplementedError('Not Implemented For RGB Image')
        org = im.fromarray(org)
        com = im.fromarray(com)
        cuave0 = im.fromarray(cuave0)
        cuave1 = im.fromarray(cuave1)
        cuave2 = im.fromarray(cuave2)
        cuave3 = im.fromarray(cuave3)

        w, h = org.size
        return {
            'cuave0':np.array(cuave0, dtype=np.float32),
            'cuave1':np.array(cuave1, dtype=np.float32),
            'cuave2':np.array(cuave2, dtype=np.float32),
            'cuave3':np.array(cuave3, dtype=np.float32),

            'com': np.array(com, dtype=np.float32),
            'com_2': np.array(com.resize((w // 2, h // 2)), dtype=np.float32),
            'com_4': np.array(com.resize((w // 4, h // 4)), dtype=np.float32),
            'org': np.array(org, dtype=np.float32),
            'org_2': np.array(org.resize((w // 2, h // 2)), dtype=np.float32),
            'org_4': np.array(org.resize((w // 4, h // 4)), dtype=np.float32)
        }


class ToTorchTensor(object):
    def __init__(self):
        pass

    def __call__(self, pair):
        image = pair['org']

        if isinstance(image, list):
            raise NotImplementedError('Not Implemented For Multiple Qs')

        isRGB = (len(image.shape) == 3 and image.shape[2] == 3)
        assert isRGB or len(image.shape) == 2

        if isRGB:
            for i in pair:
                pair[i] = np.asarray(pair[i].transpose((2, 0, 1)) / 255, dtype=np.float)
        else:
            for i in pair:
                pair[i] = np.asarray(pair[i].reshape((1,) + pair[i].shape) / 255, dtype=np.float)

        return pair


def evalPsnr(net2, v_data, fout=None, msg_header=''):
    c_psnr = 0
    p_psnr = 0
    net2.eval()
    for _, vd in enumerate(v_data, 0):
        com = vd['com'].float().cuda()
        org = vd['org'].float().cuda()
        cuave0 = vd['cuave0'].float().cuda()
        cuave1 = vd['cuave1'].float().cuda()
        cuave2 = vd['cuave2'].float().cuda()
        cuave3 = vd['cuave3'].float().cuda()
        # print(com.shape)
        out = net2(com,cuave0,cuave1,cuave2,cuave3)

        c = com.data.cpu().numpy()[0]
        c = c.reshape(c.shape[1:])
        p = out.data.cpu().numpy()[0]
        p = p.reshape(p.shape[1:])
        o = org.data.cpu().numpy()[0]
        o = o.reshape(o.shape[1:])
        c_psnr += compare_psnr(o, c)
        p_psnr += compare_psnr(o, p)
    c_psnr /= (_ + 1)
    p_psnr /= (_ + 1)
    print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr))
    if fout:
        print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr), file=fout)
    return p_psnr


def PartitionForward(net,Input,Cuave0,Cuave1,Cuave2,Cuave3):
    basesize = 128
    padding = 64
    print(Input.shape)
    _,_,row,col = Input.shape


    # print(row)
    # print(col)
    cnt = 0
    final_output = np.zeros([row,col])
    # print(final_output.shape)
    for i in range(0,row // basesize + 1):
        for j in range(0,col//basesize + 1):
            sti = 0
            stj = 0
            edi = 0
            edj = 0
            if i == 0:
                sti = 0
            else:
                sti = i*basesize - padding
            if j == 0:
                stj = 0
            else:
                stj = j * basesize - padding
            if (i + 1) * basesize + padding > row :
                edi = row
            else:
                edi = (i + 1) * basesize + padding
            if (j + 1) * basesize + padding > col :
                edj = col
            else:
                edj = (j + 1) * basesize + padding

            input = Input[:,:,sti:edi,stj:edj]
            # input = np.array([[input]])
            print("input:",input.shape)
            print("Input:",Input.shape)
            cuave0 = Cuave0[:,:,sti:edi,stj:edj]
            # cuave0 = np.array([[cuave0]])
            cuave1 = Cuave1[:,:,sti:edi,stj:edj]
            # cuave1 = np.array([[cuave1]])

            cuave2 = Cuave2[:,:,sti:edi,stj:edj]
            # cuave2 = np.array([[cuave2]])

            cuave3 = Cuave3[:,:,sti:edi,stj:edj]
            # cuave3 = np.array([[cuave3]])
            # print(sti,",",edi,",",stj,",",edj)
            # print("cuave0shape",cuave0.shape)
            output = net(input,cuave0,cuave1,cuave2,cuave3)
            output = output.cpu()
            
           

            output = output.detach().data.numpy()#*255
            # output = output.astype(np.uint8)
            output = output[0][0]
            di = 0
            dj = 0
            if sti != 0:
                sti += padding
                di = padding
            if stj != 0:
                stj += padding
                dj = padding
            if edi != row:
                edi -= padding
            if edj != col:
                edj -= padding
            # print([sti,edi,stj,edj])
            final_output[sti:edi,stj:edj] = copy.deepcopy(output[di:edi-sti+di,dj:edj-stj+dj])
            
            cnt += 1
    final_output = final_output.reshape(1,1,row,col)
    print("outshape:",final_output.shape)
    return torch.from_numpy(final_output).cuda()

def evalPsnrPartition(net2, v_data, fout=None, msg_header=''):
    c_psnr = 0
    p_psnr = 0
    net2.eval()
    for _, vd in enumerate(v_data, 0):
        com = vd['com'].float().cuda()
        org = vd['org'].float().cuda()
        cuave0 = vd['cuave0'].float().cuda()
        cuave1 = vd['cuave1'].float().cuda()
        cuave2 = vd['cuave2'].float().cuda()
        cuave3 = vd['cuave3'].float().cuda()
        # print(com.shape)
        out = PartitionForward(net2,com,cuave0,cuave1,cuave2,cuave3)

        c = com.data.cpu().numpy()[0]
        c = c.reshape(c.shape[1:])

        p = out.data.cpu().numpy()[0]

        pimage = im.fromarray(p[0]).convert("L")
        pimage.save("/mnt/ssd/wangdezhao/"+str(_)+".bmp")

        p = p.reshape(p.shape[1:])

        o = org.data.cpu().numpy()[0]
        o = o.reshape(o.shape[1:])
        tmppsnr = compare_psnr(o,c)
        print("com:",tmppsnr)
        c_psnr += tmppsnr
        tmppsnr = compare_psnr(o,p)
        print("pro:",tmppsnr)
        p_psnr += tmppsnr
        # p_psnr += compare_psnr(o, p)
        # print()
    c_psnr /= (_ + 1)
    p_psnr /= (_ + 1)
    print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr))
    if fout:
        print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr), file=fout)
    return p_psnr




def testAndSave(net, t_data, output_dir):
    ensure_exists(output_dir)

    cdir = os.path.join(output_dir, './compressed/')
    pdir = os.path.join(output_dir, './predicted/')
    ensure_exists(cdir)
    ensure_exists(pdir)

    for idx, vd in enumerate(t_data, 0):
        com = vd['com'].float().cuda()
        print(com.shape)

        _, _, out, _ = net(com)

        c = com.data.cpu().numpy()[0]
        c = c.reshape(c.shape[1:])
        p = out.data.cpu().numpy()[0]
        p = p.reshape(p.shape[1:])

        plt.imsave(os.path.join(cdir, str(idx) + '.png'), c, cmap='gray')
        plt.imsave(os.path.join(pdir, str(idx) + '.png'), p, cmap='gray')



def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

def testAndSaveColor(net, t_data, output_dir):
    ensure_exists(output_dir)

    cdir = os.path.join(output_dir, './compressed/')
    pdir = os.path.join(output_dir, './predicted/')
    ensure_exists(cdir)
    ensure_exists(pdir)

    net.eval()

    for idx, vd in enumerate(t_data, 0):
        com = vd['com'].float().cuda()
        c_img = tensor_to_PIL(com)
        c_img.save(os.path.join(cdir, str(idx) + '.png'))

        ima_r = com[0, 0, :, :].clone()
        ima_g = com[0, 1, :, :].clone()
        ima_b = com[0, 2, :, :].clone()

        #import IPython
        #IPython.embed()

#        out_ycbcr = com.clone()

        ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
        ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128
        ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128

        com[0, 0, :, :] = ima_r
        com[0, 1, :, :] = ima_cb 
        com[0, 2, :, :] = ima_cr
        
        out = net(com[:, 0:1, :, :])
        #out = net(com[:, 0:1, :, :])

#        c = com[:, 0:1, :, :].data.cpu().numpy()[0]
        c= ima_r.cpu().numpy()
        #c = c.reshape(c.shape[1:])
        p = out.data.cpu().numpy()[0]
        p = p.reshape(p.shape[1:])


        #plt.imsave(os.path.join(cdir, str(idx) + '.png'), c, cmap='gray')
        plt.imsave(os.path.join(pdir, str(idx) + '.png'), p, cmap='gray')

def PIL_to_tensor(image):
    loader = transforms.Compose([transforms.ToTensor()]) 
    image = loader(image).unsqueeze(0)
    return image.cuda()

def _ycc(r, g, b): # in (0,255) range
    r = r *255.0
    g = g *255.0
    b = b *255.0
    y = .299*r + .587*g + .114*b
    cb = 128.0 -.168736*r -.331364*g + .5*b
    cr = 128.0 +.5*r - .418688*g - .081312*b

    y = y/255.0
    cb = cb/255.0
    cr = cr/255.0

    return y, cb, cr

def _rgb(y, cb, cr):
    y = y*255.0
    cb = cb*255.0
    cr = cr*255.0

    r = y + 1.402 * (cr-128.0)
    g = y - .34414 * (cb-128.0) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128.0)

    r = r/255.0
    g = g/255.0
    b = b/255.0
    r[r<0]=0
    r[r>1]=1
    g[g<0]=0
    g[g>1]=1
    b[b<0]=0
    b[b>1]=1

    return r, g, b

def testAndSaveColorv2(net, input_dir, output_dir):
    ensure_exists(output_dir)

    files= os.listdir(input_dir)
    cdir = os.path.join(output_dir, './compressed/')
    pdir = os.path.join(output_dir, './predicted/')
    ensure_exists(cdir)
    ensure_exists(pdir)

    net.eval()

    infiles = os.listdir(input_dir)
    for infile in infiles:
        if not os.path.isdir(infile):
            print(input_dir+"/"+infile)

            com_img = im.open(input_dir+"/"+infile).convert('RGB')
            com = PIL_to_tensor(com_img)

            ima_r = com[0, 0, :, :].clone()
            ima_g = com[0, 1, :, :].clone()
            ima_b = com[0, 2, :, :].clone()

            out = com.clone()

            #import IPython
            #IPython.embed()

            ima_y, ima_cb, ima_cr = _ycc(ima_r.clone(), ima_g.clone(), ima_b.clone())
            #ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16.0/255.0
            #ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128.0/255.0
            #ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128.0/255.0

            com[0, 0, :, :] = ima_y
            com[0, 1, :, :] = (ima_cb+1)/2
            com[0, 2, :, :] = (ima_cr+1)/2


            out_y  = net(com[:, 0:1, :, :])
            out_cb  = net(com[:, 1:2, :, :])
            out_cr  = net(com[:, 2:3, :, :])

            out_cb = out_cb*2-1
            out_cr = out_cr*2-1

            out1, out2, out3 = _rgb(out_y.clone(), out_cb.clone(), out_cr.clone())

            com[0, 0, :, :] = ima_r
            com[0, 1, :, :] = ima_g
            com[0, 2, :, :] = ima_b

            out[:, 0:1, :, :] = out1
            out[:, 1:2, :, :] = out2
            out[:, 2:3, :, :] = out3

            c = tensor_to_PIL(com)
            #c = c.reshape(c.shape[1:])
            p = tensor_to_PIL(out)
            #p = p.reshape(p.shape[1:])

            c.save(os.path.join(cdir, infile[:-4] + '.png'))
            p.save(os.path.join(pdir, infile[:-4] + '.png'))
