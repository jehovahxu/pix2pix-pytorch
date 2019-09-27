#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: Jehovah
# @Date  : 18-6-7
# @Desc  : 

import os
import torchvision.utils as vutils
import option
import torch
from data2 import *
from networks import *
from pix2pix_model import *


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定gpu


if __name__ == '__main__':
    opt = option.init()
    net_G = Generator(opt.input_nc, opt.output_nc)
    net_G.load_state_dict(torch.load(opt.checkpoints+'/net_G_ins1.pth'))
    net_G.cuda()
    print "net_G loaded"
    dataloader = MyDataset(opt, '/test', isTrain=1)
    imgNum = len(dataloader)
    print len(dataloader)

    test_loader = torch.utils.data.DataLoader(dataset=dataloader, batch_size=opt.batchSize, shuffle=True, num_workers=2)

    fakeB = torch.FloatTensor(imgNum, opt.output_nc, opt.fineSize, opt.fineSize)
    A = torch.FloatTensor(imgNum, opt.output_nc, opt.fineSize, opt.fineSize)
    realB = torch.FloatTensor(imgNum, opt.output_nc, opt.fineSize, opt.fineSize)

    for i, image in enumerate(test_loader):
        imgA = image['B']
        # imgB = image['A']

        real_A = Variable(imgA.cuda())
        # real_B = Variable(imgB.cuda())

        fake_B = net_G(real_A)

        fakeB[i, :, :, :] = fake_B.data
        # realB[i, :, :, :] = real_B.data

        print ("%d.jpg generate completed" % i)

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    vutils.save_image(fakeB,
                      '%s/fakeB_8.png' % (opt.output),
                      normalize=True,
                      scale_each=True)
    # vutils.save_image(realB,
    #                   '%s/realB_8.png' % (opt.outf),
    #                   normalize=True,
    #                   scale_each=True)
