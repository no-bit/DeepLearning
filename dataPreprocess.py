import tensorflow as tf

import numpy as np
import glob, os, random
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

import scipy.io as sio
from skimage.io import imread
import cv2

from data_augment import *

tf.compat.v1.reset_default_graph()

images_folder = "./Onera Satellite Change Detection dataset - Images"
labels_folder = "./Onera Satellite Change Detection dataset - Train Labels"

train_txt = os.path.join(images_folder, "train.txt")
valid_txt = os.path.join(images_folder, "valid.txt")
test_txt = os.path.join(images_folder, "test.txt")

train_files = pd.read_csv(train_txt, sep=',', header=None).to_numpy().flatten()
valid_files = pd.read_csv(valid_txt, sep=',', header=None).to_numpy().flatten()
test_files = pd.read_csv(test_txt, sep=',', header=None).to_numpy().flatten()


selected_bands = ['B02.tif', 'B03.tif', 'B04.tif']
        
def read_whole_data(fileList, resize_to_gt = True, for_train = True):
    resize_to_gt = True
    im_1 = []
    im_2 = []
    label= []
    
    for fName in fileList:
        # ground truth
        labFolder =  os.path.join(labels_folder, fName, 'cm')
        imlist   = sorted(glob.glob(labFolder + '/*.png'))
        gt = imread( imlist[0] )
        aa = np.zeros(( gt.shape[0], gt.shape[1], 2) , gt.dtype)
        aa[:,:,0] = gt[:,:,0]
        aa[:,:,1] = 255-gt[:,:,0]
        label.append(aa)
        
        # im 1
        imFolder =  os.path.join(images_folder, fName, 'imgs_1')
        imlist   = sorted(glob.glob(imFolder + '/*.tif'))
        channelInd = 0
        if resize_to_gt:
            bb = np.zeros(( label[-1].shape[0], label[-1].shape[1], len(selected_bands)) , np.uint16)
        for targetImage in imlist:
            tarFileParts = targetImage.split('_')[-1]
            if tarFileParts in selected_bands:
                img_1 = imread( targetImage )
                if resize_to_gt:
                    if img_1.shape[0] != gt.shape[0] or img_1.shape[1] != gt.shape[1]:
                        bb[:,:,channelInd] = cv2.resize(img_1, (gt.shape[1], gt.shape[0]))
                    else:
                        bb[:,:,channelInd] = img_1
                    channelInd +=1
                else:
                    im_1.append(img_1)
                
        if resize_to_gt:
            im_1.append(bb)
        
        # im 2
        imFolder =  os.path.join(images_folder, fName, 'imgs_2')
        imlist   = sorted(glob.glob(imFolder + '/*.tif'))
        channelInd = 0
        if resize_to_gt:
            bb = np.zeros(( label[-1].shape[0], label[-1].shape[1], len(selected_bands)) , np.uint16)
        for targetImage in imlist:
            tarFileParts = targetImage.split('_')[-1]
            if tarFileParts in selected_bands:
                img_2 = imread( targetImage )
                if resize_to_gt:
                    if img_2.shape[0] != gt.shape[0] or img_2.shape[1] != gt.shape[1]:
                        bb[:,:,channelInd] = cv2.resize(img_2, (gt.shape[1], gt.shape[0]))
                    else:
                        bb[:,:,channelInd] = img_2
                    channelInd +=1
                else:
                    im_2.append(img_2)
    
        if resize_to_gt:
            im_2.append(bb)
        
    return im_1, im_2, label
    
i1, i2, l1 = read_whole_data(train_files, resize_to_gt = True, for_train = True)
vi1, vi2, vl1 = read_whole_data(valid_files, resize_to_gt = True, for_train = True)


class batcher:
    def __init__(self, width = 224, height = 224, itype = np.float32, normalize = True):
        self.batch_idx = 0
        self.w  = width
        self.h  = height
        self.dt = itype
        self.nor= normalize
        
    def buildBatches(self, listIm1, listIm2, labels, hStep=1, wStep=1, aug = ['flipv', 'fliph', 'flipflip', 'trans', 'trans1'] ):
        self.Imgs = []
        self.Lbls = []
        self.channelCount = listIm1[0].shape[2]
        for i in range( len(listIm1) ):
            typeMax = np.iinfo(listIm1[i].dtype).max
            im1 = listIm1[i].astype(self.dt)
            im2 = listIm2[i].astype(self.dt)
            
            if self.nor:
                im1 = im1/(typeMax)
                im2 = im2/(typeMax)
                
            if self.h < im1.shape[0] and self.w < im1.shape[1]:
                for hh in range(0, im1.shape[0], hStep):
                    hStart = 0
                    hStop  = 0
                    if (hh + self.h) > im1.shape[0]: 
                        hStart = im1.shape[0] - self.h
                        hStop = im1.shape[0]
                    else:
                        hStart = hh
                        hStop = hh + self.h   
                    for ww in range(0, im1.shape[1], wStep):
                        wStart = 0
                        wStop  = 0
                        if (ww + self.w) > im1.shape[1]: 
                            wStart = im1.shape[1] - self.w
                            wStop = im1.shape[1]
                        else:
                            wStart = ww
                            wStop = ww + self.w

                        zipped = np.dstack((im1[hStart:hStop, wStart:wStop, :], \
                                            im2[hStart:hStop, wStart:wStop, :] ))
                        
                        ll = labels[i][hStart:hStop, wStart:wStop, :].astype(self.dt) / 255.
                        
                        self.Imgs.append(zipped)
                        self.Lbls.append(ll)

                        if 'flipv' in aug:
                            self.Imgs.append(flipv(zipped))
                            self.Lbls.append(flipv(ll))

                        if 'fliph' in aug:
                            self.Imgs.append(fliph(zipped))
                            self.Lbls.append(fliph(ll))

                        if 'clahe' in aug:
                            self.Imgs.append(equalizeAdaptHisto_multi(zipped))
                            self.Lbls.append(ll)
                        
                        if 'rescale' in aug:
                            self.Imgs.append(rescale_intensity_multi(zipped))
                            self.Lbls.append(ll)
                        
                        if 'reflect0' in aug:
                            self.Imgs.append(paddingReflect(zipped, 50, 112, 50, 112, type =0))
                            self.Lbls.append(paddingReflect(ll, 50, 112, 50, 112, type =0))
                        
                        if 'reflect1' in aug:
                            self.Imgs.append(paddingReflect(zipped, 40, 112, 40, 112, type =1))
                            self.Lbls.append(paddingReflect(ll, 40, 112, 40, 112, type =1))
                        
                        if 'reflect2' in aug:
                            self.Imgs.append(paddingReflect(zipped, 50, 112, 50, 112, type =2))
                            self.Lbls.append(paddingReflect(ll, 50, 112, 50, 112, type =2))
                        
                        if 'reflect3' in aug:
                            self.Imgs.append(paddingReflect(zipped, 40, 112, 40, 112, type =3))
                            self.Lbls.append(paddingReflect(ll, 40, 112, 40, 112, type =3))
                        
                        if 'flipflip' in aug:
                            self.Imgs.append(fliph(flipv(zipped)))
                            self.Lbls.append(fliph(flipv(ll)))
                            
                        if 'trans' in aug:
                            self.Imgs.append(rotate_multi(zipped))
                            self.Lbls.append(rotate_multi(ll))
                            
                        if 'trans1' in aug:
                            self.Imgs.append(flipv(rotate_multi(zipped)))
                            self.Lbls.append(flipv(rotate_multi(ll)))
                            
    def giveBatch(self, batchNum ):
        self.batch_idx += batchNum
        if self.batch_idx > len(self.Imgs):
            self.batch_idx = batchNum
            
        return np.asarray( self.Imgs[self.batch_idx - batchNum : self.batch_idx] ), np.asarray( self.Lbls[self.batch_idx - batchNum:self.batch_idx] )

    def giveRandomBatch(self, batchNum ):
        datas = []
        labels= []
        indices = np.random.choice(len(self.Imgs), batchNum)   
        for idx in indices:
            imd = self.Imgs[idx]
            iml = self.Lbls[idx]
            
            datas.append(imd)
            labels.append(iml)
            
        return np.asarray( datas ), np.asarray( labels )
        
    def giveSize(self):
        return len(self.Imgs)
    
    def giveChannelCount(self):
        return self.Imgs[-1].shape[-1]
    
bat = batcher(112,112)
bat.buildBatches(i1, i2, l1, 30, 30)

batv = batcher(112,112)
batv.buildBatches(vi1, vi2, vl1, 200, 200)

del i1
del i2 
del l1
del vi1
del vi2
del vl1

a, b = batv.giveBatch(20)
for idx in range( len(a) ):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (15, 15))
    ax1.imshow( color2Gray(a[idx,:,:,0:3]) )
    ax2.imshow( color2Gray( a[idx,:,:,3:6]) )
    ax3.imshow(b[idx,:,:,0])
    ax4.imshow(b[idx,:,:,1])

"""Data augmentation includes things like randomly rotating the image, zooming in, adding a color filter etc. 
Data augmentation only happens to the training set and not on the validation/test set"""