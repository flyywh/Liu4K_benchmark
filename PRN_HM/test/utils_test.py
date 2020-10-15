import os
import torch
import numpy as np
from io import BytesIO
from PIL import Image as im
from PIL import JpegPresets
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import time

JpegPresets.presets['matlab40'] = {
    'quantization': [
        [
          20,14,13,20,30,50,64,76,
          15,15,18,24,33,73,75,69,
          18,16,20,30,50,71,86,70,
          18,21,28,36,64,109,100,78,
          23,28,46,70,85,136,129,96,
          30,44,69,80,101,130,141,115,
          61,80,98,109,129,151,150,126,
          90,115,119,123,140,125,129,124
        ], [
          20,14,13,20,30,50,64,76,
          15,15,18,24,33,73,75,69,
          18,16,20,30,50,71,86,70,
          18,21,28,36,64,109,100,78,
          23,28,46,70,85,136,129,96,
          30,44,69,80,101,130,141,115,
          61,80,98,109,129,151,150,126,
          90,115,119,123,140,125,129,124
        ]],
    'subsampling': 0
}

JpegPresets.presets['matlab30'] = {
    'quantization': [
        [
          27,18,17,27,40,66,85,101,
          20,20,23,32,43,96,100,91,
          23,22,27,40,66,95,115,93,
          23,28,37,48,85,144,133,103,
          30,37,61,93,113,181,171,128,
          40,58,91,106,134,173,188,153,
          81,106,129,144,171,201,199,168,
         120,153,158,163,186,166,171,164
        ], [
          27,18,17,27,40,66,85,101,
          20,20,23,32,43,96,100,91,
          23,22,27,40,66,95,115,93,
          23,28,37,48,85,144,133,103,
          30,37,61,93,113,181,171,128,
          40,58,91,106,134,173,188,153,
          81,106,129,144,171,201,199,168,
         120,153,158,163,186,166,171,164
        ]],
    'subsampling': 0
}

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


class ImageDir(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, root_cdir=None, root_dir1=None, root_cdir1=None,root_dir2=None, root_cdir2=None, transform=None, preload=False, is_train=False):
        self.root_dir = root_dir
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform
        self.image_list = []
        self.image_clist = []

        if root_cdir is not None:
            for r, d, filenames in os.walk(root_cdir):
                for f in filenames:
                    if f[-3:] not in ['jpg', 'png']:# or '=1' in f or '=2' in f
                        continue

            #        sf = f.split('-')[0]
            #        if is_train==False:
            #            print(os.path.join(root_dir, sf+'.jpg'))
            #            print(os.path.join(root_cdir, f))
            #        print(os.path.join(root_dir, f[:-3]+'png'))
            #        print(os.path.join(root_cdir, f))

                    self.image_list.append(os.path.join(root_dir, f[:-3]+'png'))
                    self.image_clist.append(os.path.join(root_cdir, f))

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
                    self.image_clist.append(os.path.join(root_cdir2, f[:-9]+'block.jpg'))
        
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

       # print(len(self.image_list))
       # print(len(self.image_clist))


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

        ret = {'org': tmp, 'com': tmpc}

        if self.transform:
            ret = self.transform(ret)

        ret['org_path'] = self.image_list[idx]
        ret['com_path'] = self.image_clist[idx]  
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
        h, w = image.shape[:2]
        new_h, new_w = self.l * (h // self.l), self.l * (w // self.l)

        image = image[:new_h, :new_w]
        com = com[:new_h, :new_w]
        return {'org': image, 'com': com}


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
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]
        com = com[top: top + new_h,
                left: left + new_w]
        return {'org': image, 'com': com}


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
        org = pair['org']
        com = pair['com']
        if len(org.shape) > 2:
            raise NotImplementedError('Not Implemented For RGB Image')
        org = im.fromarray(org)
        com = im.fromarray(com)
        w, h = org.size
        return {
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

def savePathConf(output_dir):
    cdir = os.path.join(output_dir, './compressed/')
    pdir = os.path.join(output_dir, './predicted/')
    odir = os.path.join(output_dir, './original/')
    ensure_exists(cdir)
    ensure_exists(pdir)
    ensure_exists(odir)
    return cdir, pdir, odir
# 
# net2, v_data, v_set, fout=None, out_dir='./tmp/', msg_header=''
#                                                                                                                                                                          
def testAndSaveCrop(net, t_data, t_set, fout=None, out_dir='./tmp/', msg_header='', pad_size=512, crop_size=32, idt = 1):                                                                                                  
    output_dir = out_dir
    ensure_exists(output_dir)                                                                                                                                             
                                                                                                                                                                          
    cdir, pdir, odir = savePathConf(output_dir)
 
    c_psnr = 0
    p_psnr = 0
    c_ssim = 0
    p_ssim = 0        
    cnt = 0
                                                                                                                                                              
    for idx, vd in enumerate(t_data, 0):                                                                                                                                  
        com = vd['com'].float().cuda()                                                                                                                                    


        cnt = cnt+1
        #if cnt<int(idt):
        #    continue
        #if cnt>int(idt):
        #    break

        com_path = vd['com_path']                                                                                                                                         
        com_path = com_path[0].split('/')                                                                                                                                 
        result_path = com_path[-1].split('.')                                                                                                                             
        result_path = result_path[0] + '.png'                                                                                                                             
                
#        print(result_path)                                                                                                                                                          
        #                                                                                                                                                                 
        # pad and crop                                                                                                                                                    
        #                                                                                                                                                                 
                                                                                                                                                                          
        val_input = com.clone()                                                                                                                                                  
        vshape = val_input.shape                                                                                                                                          
        hei = vshape[2]                                                                                                                                                   
        wid = vshape[3]                                                                                                                                                   
                                                                                                                                                                          
        o_hei = hei                                                                                                                                                       
        o_wid = wid                                                                                                                                                       
                                                                                                                                                                          
        if hei<crop_size:                                                                                                                                                 
            val_input2 = torch.zeros(vshape[0], vshape[1], crop_size, wid).cuda()
            val_input2[:,:,:hei, :] = val_input[:, :, :, :]                                                                     
            for t in range(hei, crop_size):                                                                                                                               
                val_input2[:,:,t, :] = val_input[:,:,hei-1, :]                                                                                                            
            val_input = val_input2[:, :, :, :]                                                                                                                           
            hei = crop_size                                                                                                                                               
                                                                                                                                                                          
        if wid<crop_size:                                                                                                                                                 
            val_input2 = torch.zeros(vshape[0], vshape[1], hei, crop_size).cuda()                                                                                         
            val_input2[:,:,:,:wid] = val_input[:, :, :, :]                                                                                                          
            for t in range(wid, crop_size):                                                                                                                               
                val_input2[:,:,:, t] = val_input[:,:,:,wid-1]                                                                                                             
            val_input = val_input2[:, :, :, :]                                                                                                                            
                                                                                                                                                                          
        val_result = val_input[:, :, :, :]                                                                                                                                    
        vshape_ex = val_input.shape                                                                                                                                       
                                                                                                                                                                          
        wid_ex = vshape_ex[3]                                                                                                                                             
        hei_ex = vshape_ex[2]                                                                                                                                             
                                                                                                                                                                          
        hei = hei_ex                                                                                                                                                      
        wid = wid_ex                                                                                                                                                      
                                                                                                                                                                          
        val_input_pad = torch.zeros(1, 1, hei_ex+2*pad_size, wid_ex+2*pad_size).cuda()                                                                                     
        val_input_pad[:, :, pad_size:pad_size+hei_ex,pad_size:pad_size+wid_ex] = val_input[:, :, :, :]                                                                             
                                                                                                                                                                          
                                                                                                                                                                          
        for s in range(0, pad_size):                                                                                                                                      
            for ss in range(0, pad_size):                                                                                                                                 
                val_input_pad[:,:,s,ss] = val_input_pad[:,:,pad_size+1,pad_size+1]                                                                                        
                val_input_pad[:,:,pad_size+hei_ex+s,pad_size+wid_ex+ss] = val_input_pad[:,:,pad_size+hei_ex-1,pad_size+wid_ex-1]                                          
                val_input_pad[:,:,s,pad_size+wid_ex+ss] = val_input_pad[:,:,pad_size+1,pad_size+wid_ex-1]                                                                 
                val_input_pad[:,:,pad_size+hei_ex+s,ss] = val_input_pad[:,:,pad_size+hei_ex-1,pad_size+1]                                                                 
                                                                                                                                                                          
        tmp_input_pad = torch.FloatTensor(1, 1, crop_size+2*pad_size, crop_size+2*pad_size)                                                                               
    
        for x in range(pad_size, hei+pad_size, crop_size):                                                                                                                
          if x+crop_size>=hei+pad_size:                                                                                                                                   
              break                                                                                                                                                       
                                                                                                                                                                    
          for y in range(pad_size, wid+pad_size, crop_size):                                                                                                              
            if y+crop_size>=wid+pad_size:                                                                                                                                 
                break                                                                                                                                                     
                
            #if result_path=='sailing2s_1r_0.png':
            #    import IPython
            #    IPython.embed()

            tmp_input_pad = val_input_pad[:,:,x-pad_size:x+crop_size+pad_size,y-pad_size:y+crop_size+pad_size]                                                       
            val_result_tmp = net(tmp_input_pad.cuda())                                                                                                                    
                                                                                                                                                                          
            #import IPython                                                                                                                                               
            #IPython.embed()                                                                                                                                              
                                                                                                                                                                          
            val_result[:,:,x-pad_size:x+crop_size-pad_size,y-pad_size:y+crop_size-pad_size] = val_result_tmp[:,:,pad_size:pad_size+crop_size,pad_size:pad_size+crop_size] 
                                                                                                                                                                          
          tmp_input_pad = val_input_pad.data[:,:,x-pad_size:x+crop_size+pad_size,wid_ex-crop_size:wid_ex+1*pad_size]                                                      
          val_result_tmp = net(tmp_input_pad.cuda())                                                                                                                      
                                                                                                                                                                          
          #import IPython                                                                                                                                                 
          #IPython.embed()                                                                                                                                                
                                                                                                                                                                          
          val_result[:,:,x-pad_size:x+crop_size-pad_size,wid-crop_size:wid] = val_result_tmp[:,:,pad_size:pad_size+crop_size,pad_size:pad_size+crop_size]                 
                                                                                                                                                                          
        for y in range(pad_size, wid+pad_size, crop_size):                                                                                                                
           if y+crop_size>=wid:                                                                                                                                            
              break                                                                                                                                                       
                                                                                                                                                                          
           tmp_input_pad = val_input_pad[:,:,hei_ex-crop_size:hei_ex+1*pad_size,y-pad_size:y+crop_size+pad_size]                                                           
                                                                                                                                                                          
           val_result_tmp = net(tmp_input_pad.cuda())                                                                                                                      
           val_result[:,:,hei-crop_size:hei,y-pad_size:y+crop_size-pad_size] = val_result_tmp[:,:,pad_size:pad_size+crop_size,pad_size:pad_size+crop_size]                 
                                                                                                                                                                          
        tmp_input = val_input_pad[:,:,hei_ex-crop_size:hei_ex+pad_size,wid_ex-crop_size:wid_ex+1*pad_size]                                                           
        tmp_input_pad = tmp_input                                                                                                                                         
                                                                                                                                                                          
        val_result_tmp = net(tmp_input_pad.cuda())                                                                                                                        
        val_result[:,:,hei-crop_size:hei,wid-crop_size:wid] = val_result_tmp[:,:,pad_size:pad_size+crop_size,pad_size:pad_size+crop_size]                                 
                                                                                                                                                                          
        out = val_result[:, :, :o_hei,:o_wid]                                                                                                                             
      
        org = vd['org'].float().cuda()

        c = com.data.cpu().numpy()[0]
        c = c.reshape(c.shape[1:])
        p = out.data.cpu().numpy()[0]
        p = p.reshape(p.shape[1:])
        o = org.data.cpu().numpy()[0]
        o = o.reshape(o.shape[1:])

        #import IPython
        #IPython.embed()

        #o=np.array(im.open('tmp/' + str(_) + '_org.bmp')).astype(np.float)/255.0
        #c = np.array(im.open('tmp/' + str(_) + '_com.bmp')).astype(np.float) / 255.0
        #p = np.array(im.open('tmp/' + str(_) + '_pre.bmp')).astype(np.float) / 255.0
#        print(compare_psnr(o, c))
        c_psnr += compare_psnr(o, c)
        p_psnr += compare_psnr(o, p)
        c_ssim += compare_ssim(o, c)
        p_ssim += compare_ssim(o, p)

        im.fromarray((o * 255.0).astype(np.uint8)).save(os.path.join(odir, result_path))
        im.fromarray((c * 255.0).astype(np.uint8)).save(os.path.join(cdir, result_path))
        im.fromarray((p * 255.0).astype(np.uint8)).save(os.path.join(pdir, result_path))
        del val_input_pad
        del val_result
        del val_input
        del out
        #del val_result_tmp
        #del tmp_input
        #del tmp_input_pad

    c_psnr = c_psnr/cnt
    p_psnr = p_psnr/cnt
    c_ssim = c_ssim/cnt
    p_ssim = p_ssim/cnt

    print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr))
    print(msg_header, 'SSIM: %.5f -> %.5f\n' % (c_ssim, p_ssim))

    if fout:
        print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr), file=fout)
    return p_psnr
                                                                                                                                                                    

def evalPsnr(net2, v_data, v_set, fout=None, out_dir='./tmp/', msg_header=''):
    c_psnr = 0
    p_psnr = 0
    c_ssim = 0
    p_ssim = 0
    running_time = 0
    net2.eval()

    ensure_exists(out_dir)
    #import IPython
    #IPython.embed()

    cnt = 0
    for _, vd in enumerate(v_data, 0):
        com = vd['com'].float().cuda()
        org = vd['org'].float().cuda()
        #IPython.embed()

        start_time = time.time()
        out = net2(com)
        end_time = time.time()
        running_time += end_time - start_time

        #print(end_time - start_time)
        c = com.data.cpu().numpy()[0]
        c = c.reshape(c.shape[1:])
        p = out.data.cpu().numpy()[0]
        p = p.reshape(p.shape[1:])
        o = org.data.cpu().numpy()[0]
        o = o.reshape(o.shape[1:])

        #o=np.array(im.open('tmp/' + str(_) + '_org.bmp')).astype(np.float)/255.0
        #c = np.array(im.open('tmp/' + str(_) + '_com.bmp')).astype(np.float) / 255.0
        #p = np.array(im.open('tmp/' + str(_) + '_pre.bmp')).astype(np.float) / 255.0
        c_psnr += compare_psnr(o, c)
        p_psnr += compare_psnr(o, p)
        c_ssim += compare_ssim(o, c)
        p_ssim += compare_ssim(o, p)

        im.fromarray((o * 255.0).astype(np.uint8)).save(out_dir + str(_) + '_org.bmp')
        im.fromarray((c * 255.0).astype(np.uint8)).save(out_dir + str(_) + '_com.bmp')
        im.fromarray((p * 255.0).astype(np.uint8)).save(out_dir + str(_) + '_pre.bmp')
        cnt = cnt + 1

    c_psnr /= cnt
    p_psnr /= cnt
    c_ssim /= cnt
    p_ssim /= cnt
    running_time /= cnt

    print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr))
    print(msg_header, 'SSIM: %.5f -> %.5f\n' % (c_ssim, p_ssim))
    print(msg_header, 'Running Time: %.5f\n' % (running_time))

    if fout:
        print(msg_header, 'PSNR: %.5f -> %.5f\n' % (c_psnr, p_psnr), file=fout)
        print(msg_header, 'Running Time: %.5f\n' % (running_time), file=fout)
    return p_psnr


def testAndSave(net, t_data, output_dir):
    ensure_exists(output_dir)

    cdir = os.path.join(output_dir, './compressed/')
    pdir = os.path.join(output_dir, './predicted/')
    ensure_exists(cdir)
    ensure_exists(pdir)

    for idx, vd in enumerate(t_data, 0):
        com = vd['com'].float().cuda()
        _, _, out, _ = net((0, 0, com))

        c = com.data.cpu().numpy()[0]
        c = c.reshape(c.shape[1:])
        p = out.data.cpu().numpy()[0]
        p = p.reshape(p.shape[1:])

        plt.imsave(os.path.join(cdir, str(idx) + '.png'), c, cmap='gray')
        plt.imsave(os.path.join(pdir, str(idx) + '.png'), p, cmap='gray')

