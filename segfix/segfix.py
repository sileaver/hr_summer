#!/usr/bin/env python3
import cv2
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.nn.functional as F
import numpy as np
from PIL import Image
import scipy.io as io
import subprocess
import multiprocessing.pool as mpp
import matplotlib.pyplot as plt
DATA_ROOT = subprocess.check_output(
    ['bash', '-c', "source config.profile; echo $DATA_ROOT"]
).decode().strip()

import os
import sys
import argparse
import os.path as osp

script_path = osp.abspath(osp.join(osp.dirname(__file__)))
os.chdir(osp.join(script_path, '..', '..'))
sys.path.insert(0, os.getcwd())
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

class LabelTransformer:

    label_list = [7, 8, 11, 12, 13, 17, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    @staticmethod
    def encode(labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.int) * 255
        for i in range(len(LabelTransformer.label_list)):
            class_id = LabelTransformer.label_list[i]
            encoded_labelmap[labelmap == class_id] = i

        return encoded_labelmap

    @staticmethod
    def decode(labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        decoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.uint8) * 255
        for i in range(len(LabelTransformer.label_list)):
            class_id = i
            decoded_labelmap[labelmap ==
                             class_id] = LabelTransformer.label_list[i]

        return decoded_labelmap


def gen_coord_map(H, W):
    coord_vecs = [fluid.layers.range(0,length,1, dtype="float") for length in (H, W)]
    coord_h, coord_w = paddle.meshgrid(coord_vecs)
    return coord_h, coord_w

def shift(x, offset):
    """
    x: h x w
    offset: 2 x h x w
    """
    h, w = x.shape
    x = fluid.dygraph.to_variable(x).unsqueeze(0)
    offset = fluid.dygraph.to_variable(offset).unsqueeze(0)
    coord_map = gen_coord_map(h, w)
    norm_factor = paddle.to_tensor([(w-1)/2, (h-1)/2],place=paddle.CPUPlace())
    grid_h = offset[:, 0]+coord_map[0]
    grid_w = offset[:, 1]+coord_map[1]
    grid = fluid.layers.stack([grid_w, grid_h],axis=-1) / norm_factor - 1
    # x = F.grid_sample(x.unsqueeze(1).float(), grid, padding_mode='border', mode='bilinear').paddle.squeeze().numpy()
    x = paddle.cast(x, "float32")
    x = F.grid_sample(x.unsqueeze(1), grid, padding_mode='border', mode='bilinear').squeeze().numpy()
    x = np.round(x)
    return x.astype(np.uint8)

def get_offset(basename):
    return io.loadmat(osp.join(offset_dir, basename+'.mat'))['mat']\
        .astype(np.float32).transpose(2, 0, 1) * args.scale

def process(basename):
    if len(basename)!=0:
        infile = osp.join(in_label_dir, basename + '.png')
        outfile = osp.join(out_label_dir, basename + '.png')
        print(infile)
        input_label_map = np.array(Image.open(infile).convert('P'))
        # np.save("/home/aistudio/sa/a.npy",input_label_map)
        
        # input_label_map = LabelTransformer.encode(input_label_map)
        plt.plot(input_label_map)
        plt.imsave("/home/aistudio/sa/b",input_label_map)
        offset_map = get_offset(basename)
        output_label_map = shift(input_label_map, offset_map)
        # output_label_map = LabelTransformer.decode(output_label_map)
        Image.fromarray(output_label_map).save(outfile)
        print('Writing', outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--offset')
    parser.add_argument('--out')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--scale', type=float, default=2)
    args = parser.parse_args()

    if args.offset is None:
        if args.split == 'val':
            offset_dir = osp.join(DATA_ROOT, 'cityscapes', 'val', 'offset_pred', 'semantic', 'offset_hrnext')
        else:
            offset_dir = osp.join(DATA_ROOT, 'cityscapes', 'test_offset', 'semantic', 'offset_hrnext')
    else:
        offset_dir = args.offset

    in_label_dir = args.input
    if args.out is None:
        if '/label' in in_label_dir:
            out_label_dir = in_label_dir.replace('/label', '/label_w_segfix')
        else:
            out_label_dir = osp.join('/home/aistudio/contrast_seg_paddle/output', 'label_w_segfix')
    else:
        out_label_dir = args.out
    print('Saving to', out_label_dir)

    os.makedirs(out_label_dir, exist_ok=True)
    input_args = [fn.rpartition('.')[0] for fn in os.listdir(in_label_dir)]
    print(len(input_args), 'files in total.')
    for basename in input_args:
        process(basename)
    # mpp.Pool().map(process, input_args)

    if args.split == 'val':
        os.system('{}  /home/aistudio/contrast_seg_paddle/lib/metrics/cityscapes_evaluator.py --gt_dir {}/cityscapes/val/label --pred_dir {}'.format(sys.executable, DATA_ROOT, out_label_dir))