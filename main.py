from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import config as cf

import os
import sys
import time
import argparse
import datetime
import random

from torch.autograd import Variable
from data_loader import ImageFolder
from networks import *

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--shuffle', type=bool, default=True, action='store_false', help='shuffle the order of data')
parser.add_argument('--dataset', required=True, help='mnist | cifar10')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the width & height of the input image')

parser.add_argument('--nz', type=int, default=100, help='size of latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')

parser.add_argument('--nEpochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=2e-4')
parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam optimizer, default=0.5')

parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--outf', default='./checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

######################### Global Variables
if(opt.dataset == 'cifar10'):
    nc = 3
elif(opt.dataset == 'cell'):
    nc = 3
elif(opt.dataset == 'mnist'):
    nc = 1
else:
    print("Error : Dataset must be one of \'mnist | cifar10 | cell\'")
    sys.exit(1)

noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label, fake_label = 1, 0

noise = Variable(noise)
real = Variable(real)
label = Variable(label)


def train_shuffle(loader, netD, netG, criterion, optimizerD, optimizerG):
    for epoch in range(1, opt.nEpochs+1):
        for i, (item) in enumerate(loader): # We don't need the class label information
            ''' item = [image(=real)-> tensor, label->tensor] list'''
            ######################### fDx : Gradient of Discriminator
            netD.zero_grad()
            ''' shuffle 없이 original 학습 '''
            # train with real data
            with torch.no_grad():
                real.resize_(item[0].size()).copy_(item[0])
                label.resize_(item[1].size(0)).fill_(real_label)

            output = netD(real) # Forward propagation, this should result in '1'
            errD_real = 0.5 * torch.mean((output-label)**2) # criterion(output, label)
            errD_real.backward()

            # train with fake data
            with torch.no_grad():
                label.fill_(fake_label)
                noise.resize_(item[0].size(0), opt.nz, 1, 1) # batch size가 추가되도록 resize
                noise.normal_(0, 1)

            fake = netG(noise) # Create fake image
            output = netD(fake.detach()) # Forward propagation for fake, this should result in '0'
            errD_fake = 0.5 * torch.mean((output-label)**2) # criterion(output, label)
            errD_fake.backward()
            #### Appendix ####
            #### var.detach() = Variable(var.data), difference in computing trigger

            errD = errD_fake + errD_real
            optimizerD.step()

            ######################### fGx : Gradient of Generator
            netG.zero_grad()
            with torch.no_grad():
                label.fill_(real_label)
            output = netD(fake) # Forward propagation of generated image, this should result in '1'
            errG = 0.5 * torch.mean((output - label)**2) # criterion(output, label)
            errG.backward()
            optimizerG.step()

            ######################### LOG
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d] Loss(D): %.4f Loss(G): %.4f '
                    %(epoch, opt.nEpochs, i, len(loader), errD.item(), errG.item()))
            sys.stdout.flush()

        ######################### Visualize
        if(i%1 == 0):
            print(": Saving current results...")
            vutils.save_image(
                fake.data,
                '%s/fake_samples_%03d.png' %(opt.outf, epoch),
                normalize=True
            )

    return


def train_ordered(loader, netD, netG, criterion, optimizerD, optimizerG):
    '''
        training the data ordered by target
            ex) MNIST data : train 0 ~ 9 data by order.
    '''
    ordered_label = [Variable(torch.FloatTensor(opt.batchSize)).fill_(i) for i in range(10)] # i번 index에 i가 오도록
    turn = 0 # 이번에 학습할 라벨로 {(turn%10)}해서 사용함. 0부터 시작해서 1,2,3,...,9 반복
    mydata = [torch.tensor([]) for i in range(10)] # label=0~9 인 각 데이터들을 따로 저장할 공간
    for epoch in range(1, opt.nEpochs+1):
        for i, (item) in enumerate(loader): # We don't need the class label information
            ''' item = [image(=real)-> tensor, label->tensor] list'''
            ######################### fDx : Gradient of Discriminator
            netD.zero_grad()

            ''' 0~9 순서대로 학습시켜보자 '''
            ''' 주의: loader에서 shuffle=false로 지정해주면서 이전에 나왔던 이미지가 다시 나와 저장되는 것을 방지했음. '''
            indices = [(item[1] == i).nonzero().flatten() for i in range(10)] # 이번 배치에서 label별 인덱스 추출
            for i in range(10):
                mydata[i] = torch.cat([mydata[i],item[0][indices[i]]],dim=0) # 각 라벨 별 데이터 concat하여 64개 뽑아낼 수 있을 때까지 이어붙임

            # 64개 채워진 애들 학습
            is_train = 1
            while is_train :
                if mydata[turn%10].size(0) >= opt.batchSize : # 64개 이상 채워진 경우
                    item = [mydata[turn%10][:64],ordered_label[turn%10]]
                    mydata[turn%10] = mydata[turn%10][64:] 
                else: # 아직 64개 안 채워진 경우
                    is_train = 0
                    continue
                
                turn += 1
                # train with real data
                with torch.no_grad():
                    real.resize_(item[0].size()).copy_(item[0])
                    label.resize_(item[1].size(0)).fill_(real_label)

                output = netD(real) # Forward propagation, this should result in '1'
                errD_real = 0.5 * torch.mean((output-label)**2) # criterion(output, label)
                errD_real.backward()

                # train with fake data
                with torch.no_grad():
                    label.fill_(fake_label)
                    noise.resize_(item[0].size(0), opt.nz, 1, 1) # batch size가 추가되도록 resize
                    noise.normal_(0, 1)

                fake = netG(noise) # Create fake image
                output = netD(fake.detach()) # Forward propagation for fake, this should result in '0'
                errD_fake = 0.5 * torch.mean((output-label)**2) # criterion(output, label)
                errD_fake.backward()
                #### Appendix ####
                #### var.detach() = Variable(var.data), difference in computing trigger

                errD = errD_fake + errD_real
                optimizerD.step()

                ######################### fGx : Gradient of Generator
                netG.zero_grad()
                with torch.no_grad():
                    label.fill_(real_label)
                output = netD(fake) # Forward propagation of generated image, this should result in '1'
                errG = 0.5 * torch.mean((output - label)**2) # criterion(output, label)
                errG.backward()
                optimizerG.step()

                ######################### LOG
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d] Loss(D): %.4f Loss(G): %.4f '
                        %(epoch, opt.nEpochs, i, len(loader), errD.item(), errG.item()))
                sys.stdout.flush()
            '''/ 0~9 순서대로 학습시켜보자 '''
        
        ######################### Visualize
        if(i%1 == 0):
            print(": Saving current results...")
            vutils.save_image(
                fake.data,
                '%s/fake_samples_%03d.png' %(opt.outf, epoch),
                normalize=True
            )

    return


def main():
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()

    ######################### Data Preperation
    print("\n[Phase 1] : Data Preperation")
    print("| Preparing %s dataset..." %(opt.dataset))

    dset_transforms = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[opt.dataset], cf.std[opt.dataset])
    ])

    if (opt.dataset == 'cifar10'):
        dataset = dset.CIFAR10(
            root='./data/cifar10/',
            download=True,
            transform=dset_transforms
        )
    elif (opt.dataset == 'mnist'):
        dataset = dset.MNIST(
            root='./data/mnist/',
            download=True,
            transform=dset_transforms
        )
    elif (opt.dataset == 'cell') :
        dataset = ImageFolder(
            root='./data/cell/',
            transform=dset_transforms
        )
    else:
        print("Error | Dataset must be one of mnist | cifar10")
        sys.exit(1)

    print("| Consisting data loader for %s..." %(opt.dataset))
    loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = opt.batchSize,
        shuffle = False
    )

    ######################### Model Setup
    print("\n[Phase 2] : Model Setup")
    ndf = opt.ndf
    ngf = opt.ngf

    print("| Consisting Discriminator with ndf=%d" %ndf)
    print("| Consisting Generator with z=%d" %opt.nz)
    netD = Discriminator(ndf, nc)
    netG = Generator(opt.nz, ngf, nc)

    if(use_cuda):
        netD.cuda()
        netG.cuda()
        cudnn.benchmark = True

    ######################### Loss & Optimizer
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))

    if(use_cuda):
        noise = noise.cuda()
        real = real.cuda()
        label = label.cuda()

    ######################### Training Stage
    print("\n[Phase 4] : Train model")
    if opt.shuffle:
        train_shuffle(loader, netD, netG, criterion, optimizerD, optimizerG)
    else:
        train_ordered(loader, netD, netG, criterion, optimizerD, optimizerG)

    ######################### Save model
    torch.save(netG.state_dict(), '%s/netG.pth' %(opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' %(opt.outf))


if __name__ == '__main__':
    main()