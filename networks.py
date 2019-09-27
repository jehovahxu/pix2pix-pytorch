#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : networks.py
# @Author: Jehovah
# @Date  : 18-6-4
# @Desc  : 

import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(Generator, self).__init__()
        #256*256
        self.en1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.en2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True)
        )
        self.en3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.en4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.en8 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8,ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(True)
        )
        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),

            nn.ReLU(True)
        )
        self.de6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.de7 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.de8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc,
                               kernel_size=4, stride=2,
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        #encoder
        out_en1 = self.en1(x)
        out_en2 = self.en2(out_en1)
        out_en3 = self.en3(out_en2)
        out_en4 = self.en4(out_en3)
        out_en5 = self.en5(out_en4)
        out_en6 = self.en6(out_en5)
        out_en7 = self.en7(out_en6)
        out_en8 = self.en8(out_en7)
        #decoder
        out_de1 = self.de1(out_en8)
        out_de1 = torch.cat((out_de1, out_en7), 1)
        out_de2 = self.de2(out_de1)
        out_de2 = torch.cat((out_de2, out_en6), 1)
        out_de3 = self.de3(out_de2)
        out_de3 = torch.cat((out_de3, out_en5), 1)
        out_de4 = self.de4(out_de3)
        out_de4 = torch.cat((out_de4, out_en4), 1)
        out_de5 = self.de5(out_de4)
        out_de5 = torch.cat((out_de5, out_en3), 1)
        out_de6 = self.de6(out_de5)
        out_de6 = torch.cat((out_de6, out_en2), 1)
        out_de7 = self.de7(out_de6)
        out_de7 = torch.cat((out_de7, out_en1), 1)
        out_de8 = self.de8(out_de7)
        return out_de8


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.cov5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_cov1 = self.cov1(x)
        out_cov2 = self.cov2(out_cov1)
        out_cov3 = self.cov3(out_cov2)
        out_cov4 = self.cov4(out_cov3)
        out = self.cov5(out_cov4)
        return out


