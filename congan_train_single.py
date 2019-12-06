import os, sys
sys.path.append(os.getcwd())

import time
import functools
import argparse

import numpy as np
#import sklearn.datasets

import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

import pdb
#import gpustat

#from models.conwgan import *
#from models.dcgan import *
#from models.dcganv2 import *
from models.dcganv3 import *
#from models.dcganv4 import *

from models.constrainer import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer

import torch.nn.init as init

import models.HDF5Dataset as H





EXP = 'WGAN-GP_v1-10-50GeV'

BATCH_SIZE = 200 # Batch size. Must be a multiple of N_GPUS

#dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE,
#                                        shuffle=True, num_workers=4)



TRAINING_CLASS = [10, 50]
VAL_CLASS = [10, 50]
NUM_CLASSES = 2




RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration 
OUTPUT_PATH = '/beegfs/desy/user/eren/improved-wgan-pytorch/output/WGAN-GP_v1/' # output path where result (.e.g drawing images, cost, chart) will be stored
# MODE = 'wgan-gp'
NDF = 128 # Model dimensionality (critic)
NGF = 50 # Model dimensionality (generator) 
DIM = 30 
LATENT = 100
CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
END_ITER = 100000 # How many iterations to train for
LAMBDA = 5 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 30*30*1 # Number of pixels in each iamge
#ACGAN_SCALE = 1.0 # How to scale the critic's ACGAN loss relative to WGAN loss
KAPPA = 0.05 # How to scale generator's ACGAN loss relative to WGAN loss

GEN = 'netG_itrs_7999.pth'
CRIT = 'netD_itrs_7999.pth'
REG = 'netE_itrs_7999.pth'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)    


def calc_gradient_penalty(netD, real_data, real_label, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 1, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 1, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)   

    disc_interpolates = netD(interpolates.float(), real_label.float())

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



def gen_rand_noise_with_label(label=None):
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT)) * label    
    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise



cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


if RESTORE_MODE:
    aD = DCGAN_D(DIM,1,NDF)
    aG = DCGAN_G(1,NGF,LATENT)
    aE = energyRegressor(1)
    aG.load_state_dict(torch.load(OUTPUT_PATH+GEN, map_location=torch.device(device)))
    aD.load_state_dict(torch.load(OUTPUT_PATH+CRIT, map_location=torch.device(device)))
    aE.load_state_dict(torch.load(OUTPUT_PATH+REG, map_location=torch.device(device)))

else:
    aD = DCGAN_D(DIM,1,NDF)
    aG = DCGAN_G(1,NGF,LATENT)
    aE = energyRegressor(1)

    aG.apply(weights_init)
    aD.apply(weights_init)
    aE.apply(weights_init)


netD_total_params = sum(p.numel() for p in aD.parameters() if p.requires_grad)
netG_total_params = sum(p.numel() for p in aG.parameters() if p.requires_grad)
print (aD, netD_total_params)
print (aG, netG_total_params)


LR_g = 1e-5
LR_c = 1e-5
LR_reg = 1e-6
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR_g, betas=(0.5, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR_c, betas=(0.5, 0.9))

#optimizer_g = torch.optim.RMSprop(aG.parameters(), lr=LR_g)
#optimizer_d = torch.optim.RMSprop(aD.parameters(), lr=LR_c)
optimizer_e = torch.optim.Adam(aE.parameters(), lr=LR_reg, betas=(0,0.9))

e_criterion = nn.MSELoss() # for energy regressor training
gen_criterion = nn.MSELoss()  ## for generator training

one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
aE = aE.to(device)
one = one.to(device)
mone = mone.to(device)

writer = SummaryWriter()
#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    #writer = SummaryWriter()
   
    #load data and make it iterable
    path = 'data/gamma-fullG-fixed50-10GeV.hdf5'
    data = H.HDF5Dataset(path, '30x30')
    energies = data['energy'][:].reshape(len(data['energy']))
    layers = data['layers'][:].sum(axis=1)

    training_dataset = tuple(zip(layers, energies))
    
    dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=4, drop_last=True)
    dataiter = iter(dataloader)


    for iteration in range(START_ITER, END_ITER):
        
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        
        for e in aE.parameters():  # reset requires_grad (constrainer)
            e.requires_grad_(True)  # they are set to False below in training G

        #if iteration < 300:
        #    CRITIC_ITERS = 50
        
        for i in range(CRITIC_ITERS):
            #print("Critic iter: " + str(i))
            
            
            aD.zero_grad()
            aE.zero_grad()

            # gen fake data and load real data
            inc_energy_label = [50.0, 50.0]
            f_label = np.random.choice(inc_energy_label, (BATCH_SIZE,1), p=[0.5, 0.5])

         
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT))     
            noise = torch.from_numpy(noise).float()
            noise = noise.to(device)

            y_label = torch.from_numpy(f_label).float()
            y_label = y_label.to(device)

            
            batch = next(dataiter, None)

            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()

            real_label = batch[1] ## energy label
            real_label = real_label.unsqueeze(-1)  ## transform to [Bs, 1 ]
            real_label = real_label.to(device)
            real_label.requires_grad_(True)

            y_label = real_label

            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev, y_label).detach()
                
            
            

            real_data = batch[0] # 30x30 calo layers
            real_data = real_data.unsqueeze(1)  ## transform to [Bs, 1, 30 , 30 ]
            real_data = real_data.to(device)
            real_data.requires_grad_(True)

            #real_label = batch[1] ## energy label
            #real_label = real_label.unsqueeze(-1)  ## transform to [Bs, 1 ]
            #real_label = real_label.to(device)
            

            #### supervised-training for energy regressor!

            output = aE(real_data.float())
            e_loss = e_criterion(output, real_label).mean()
            e_loss.backward()
            optimizer_e.step()
        
            ######

            

            # train with real data
            disc_real = aD(real_data.float(), real_label.float())
            disc_real = disc_real.mean()


            # train with fake data
            disc_fake = aD(fake_data, y_label)
            disc_fake = disc_fake.mean()

          
            # train with interpolated data
            
            gradient_penalty = calc_gradient_penalty(aD, real_data, real_label, fake_data)
            

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            #disc_cost = disc_cost.mean()
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                writer.add_scalar('data/wasserstein_distance',  w_dist.mean(), iteration)
                writer.add_scalar('data/e_loss', e_loss, iteration)
                
        
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D
        
        for c in aE.parameters():
            c.requires_grad_(False)  # freeze C

        gen_cost = None
        for i in range(GENER_ITERS):
            
            aG.zero_grad()
            
            inc_energy_label = [10.0, 50.0]
            f_label = np.random.choice(inc_energy_label, (BATCH_SIZE,1), p=[0.5, 0.5])

            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT))     
            noise = torch.from_numpy(noise).float()
            noise = noise.to(device)

            y_label = torch.from_numpy(f_label).float()
            y_label = y_label.to(device)
            
            noise.requires_grad_(True)
            
            
            fake_data = aG(noise, y_label)
                       
       
            gen_cost = aD(fake_data.float(), y_label)
            c = aE(fake_data)
            
            aux_errG = gen_criterion(c, y_label).mean()
            gen_cost = -gen_cost.mean()
            g_cost = KAPPA*aux_errG + gen_cost
            g_cost.backward()
        
        optimizer_g.step()
        #end = timer()
        #print(f'---train G elapsed time: {end - start}')

        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)
        writer.add_scalar('data/e_loss_aG', aux_errG, iteration)
        #if iteration %200==199:
        #   paramsG = aG.named_parameters()
        #   for name, pG in paramsG:
        #       writer.add_histogram('G.' + name, pG.clone().data.cpu().numpy(), iteration)
	    #----------------------Generate images-----------------

        #lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        #lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        #lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        #lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        #print ('iteration: {}, critic loss: {}'.format(iteration, disc_cost.cpu().data.numpy()) )
        if iteration % 500==499 or iteration == 1 :
            print ('iteration: {}, critic loss: {}'.format(iteration, disc_cost.cpu().data.numpy()) )
            torch.save(aG.state_dict(), 'output/{0}/netG_itrs_{1}.pth'.format(EXP, iteration))
            torch.save(aD.state_dict(), 'output/{0}/netD_itrs_{1}.pth'.format(EXP, iteration))
            torch.save(aE.state_dict(), 'output/{0}/netE_itrs_{1}.pth'.format(EXP, iteration))
        

            
    

train()


