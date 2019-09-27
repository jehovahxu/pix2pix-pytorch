#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : option.py
# @Author: Jehovah
# @Date  : 18-6-4
# @Desc  : 


import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--dataroot', required=True,
                        help="path to images (should have subfolders trainA, trainB, valA, valB, etc)")
    parser.add_argument('--gpuid', type=str, default='0', help='which gpu to use')
    parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--bata', type=int, default=0.5, help='momentum parameters bata1')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='with batchSize=1 equivalent to instance normalization.')
    parser.add_argument('--niter', type=int, default=600, help='number of epochs to train for')
    parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
    parser.add_argument('--sample', type=str, default='./sample', help='models are saved here')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='image are saved here')
    parser.add_argument('--output', default='./output', help='folder to output images ')
    parser.add_argument('--datalist', default='files/list_train.txt')
    parser.add_argument('--pre_net', default='/home/kejia/PycharmProjects/pix2pix_xxx/experiment/')
    opt = parser.parse_args()
    return opt
