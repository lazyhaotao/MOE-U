#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../project/')
from tools import *

DATASET = 'KSC' # Dataset name
FOLDER = ' ' # the dataset folder
N_RUNS = 10 # the runing times of the experiments
SAMPLE_SIZE = 10 # training samples per class
SAMPLING_MODE = 'fixed_withone' # fixed number for each class
def getSplits():
    # datasets prepare
    ''' img: array 3D; gt: array 2D;'''
    #img: Normalized hyperspectral image
    #gt：Pixel labels, 2D array
    #LABEL_VALUES：One-dimensional array, the names corresponding to the label numbers
    #IGNORED_LABELS:[0]
    #RGB_BANDS：The number of channels where the RGB band is located
    #platte：none
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
    # Number of classes + unidefind label
    N_CLASSES = len(LABEL_VALUES) - 1
    # Number of bands
    N_BANDS = img.shape[-1]
    # run the experiment several times
    for run in range(N_RUNS):
        #tain_gt:Training sample labels, select a certain number for each category, the same size as gt, the labels of training samples are non-0, and the labels of test samples are 0
        #test_gt：The test sample label is the same size as gt, the test sample label is non-0, and the training sample label is 0
        train_gt, test_gt = sample_gt(gt, SAMPLE_SIZE, mode=SAMPLING_MODE)
        save_sample(train_gt, test_gt, DATASET, SAMPLE_SIZE, run)
    
if __name__ == '__main__':
    getSplits()
        
