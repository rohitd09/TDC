import multiprocessing
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

batchSize = 64
imageSize = 64

transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
nc = 3

dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def copyGenerator():
    layer_num = 0
    for param in netG.parameters():
        if (rank == 0):
            data = param.data.numpy().copy()
        else:
            data = None
        data = comm.bcast(data, root = 0)
        if (rank != 0):
            param.data = torch.from_numpy(data)

        layer_num += 1

def multiDiscriminators():
    if (rank != 0):
        layer_num = 0
        for param in netD.parameters():
            outdata = param.data.numpy().copy()
            indata = None

            if (rank != size - 1):
                comm.send(outdata, dest=rank + 1, tag=1)
            if (rank != 1):
                indata = comm.recv(source = rank-1, tag=1)

            if (rank == size - 1):
                comm.send(outdata, dest=1, tag=2)
            if (rank == 1):
                indata = comm.recv(source = size - 1, tag=2)
            param.data = torch.from_numpy(indata)
            layer_num += 1

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = G()
netG.apply(weights_init)

class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

def main():
    netD = D()
    netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    for epoch in range(26):

        if (epoch % 2 == 0):
            multiDiscriminators()

        for i, data in enumerate(dataloader, 775):
            
            netD.zero_grad()
            real, _ = data
            input = Variable(real)
            target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            errD_real = criterion(output, target)
            
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]))
            output = netD(fake.detach())
            errD_fake = criterion(output, target)
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0]))
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.item(), errG.item()))
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
                fake = netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
        
        print(f"End of epoch number: {epoch}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()