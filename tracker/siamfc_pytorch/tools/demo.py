from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    currentDir = os.getcwd()

    parentDir = os.path.abspath( os.path.join(currentDir, os.pardir ) )
    print(parentDir)
    seq_dir = os.path.join(parentDir, 'data', 'OTB', 'Crossing')
    print(seq_dir)
    #seq_dir = os.path.expanduser('~/data/OTB/Crossing/')
    img_files = sorted(glob.glob(seq_dir + '\img\****.jpg'))
    print(glob.glob(seq_dir + '\img\****.jpg'))
    anno = np.loadtxt(seq_dir + '\Crossing_groundtruth_rect2.txt')
    print(seq_dir + '\Crossing_groundtruth_rect.txt2')
    
    net_path = 'pretrained\siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    x = tracker.track(img_files, anno, visualize=True)
    print(x)