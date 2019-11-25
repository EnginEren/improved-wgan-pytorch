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
from models.dcganv2 import *

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



## Load data and make them iterable

path = 'data/gamma-fullG-fixed50-10GeV.hdf5'

data = H.HDF5Dataset(path, '30x30')
energies = data['energy'][:].reshape(len(data['energy']))
layers = data['layers'][:].sum(axis=1)

training_dataset = tuple(zip(layers, energies))

EXP = 'testGP-ndf64v2'

BATCH_SIZE = 100 # Batch size. Must be a multiple of N_GPUS

dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=8)



TRAINING_CLASS = [10, 50]
VAL_CLASS = [10, 50]
NUM_CLASSES = 2




RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration 
OUTPUT_PATH = '/beegfs/desy/user/eren/improved-wgan-pytorch/output/testGP/' # output path where result (.e.g drawing images, cost, chart) will be stored
# MODE = 'wgan-gp'
NDF = 64 # Model dimensionality (critic)
NGF = 64 # Model dimensionality (generator) 
DIM = 30 
LATENT = 500
CRITIC_ITERS = 10 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
END_ITER = 100000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 30*30*1 # Number of pixels in each iamge
ACGAN_SCALE = 1.0 # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1.0 # How to scale generator's ACGAN loss relative to WGAN loss

GEN = ''
CRIT = ''



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
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

    disc_interpolates, _ = netD(interpolates.float(), real_label.float())

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
    aG = torch.load(OUTPUT_PATH + GEN)
    aD = torch.load(OUTPUT_PATH + CRIT)
else:
    aD = DCGAN_D(DIM,1,NDF,NUM_CLASSES)
    aG = DCGAN_G(1,NGF,LATENT)
    
    aG.apply(weights_init)
    aD.apply(weights_init)

LR = 1e-4
optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0,0.9))

aux_criterion = nn.CrossEntropyLoss() # nn.NLLLoss()

one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)

writer = SummaryWriter()
#Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    #writer = SummaryWriter()
    #dataloader = load_data(DATA_DIR, TRAINING_CLASS)
    #dataiter = iter(dataloader)

    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        #print("Iter: " + str(iteration))
        start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):
            #print("Generator iters: " + str(i))
            aG.zero_grad()
            inc_energy_label = [10.0, 50.0]
            f_label = np.random.choice(inc_energy_label, (BATCH_SIZE,1), p=[0.5, 0.5])
            noise = gen_rand_noise_with_label(f_label)
            noise.requires_grad_(True)
            fake_data = aG(noise)
                       
            aux_label = torch.from_numpy(f_label).long()
            
            aux_label = aux_label.to(device)

            gen_cost, gen_aux_output = aD(fake_data, aux_label.float())
            
            
            aux_label = aux_label.view(-1)

            aux_label = (aux_label - 10) / 40  ## transform labels --> 0 and 1 
            aux_errG = aux_criterion(gen_aux_output, aux_label).mean()
            gen_cost = -1.0 * gen_cost.mean()
            g_cost = ACGAN_SCALE_G*aux_errG + gen_cost
            g_cost.backward()
        
        optimizer_g.step()
        #end = timer()
        #print(f'---train G elapsed time: {end - start}')
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            #print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            inc_energy_label = [10, 50]
            f_label = np.random.choice(inc_energy_label, (BATCH_SIZE,1), p=[0.5, 0.5])
            noise = gen_rand_noise_with_label(f_label)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            #end = timer(); print(f'---gen G elapsed time: {end-start}')
            start = timer()
            
            
            dataiter = iter(dataloader)
            batch = dataiter.next()
            
            real_data = batch[0] # 30x30 calo layers
            real_data = real_data.unsqueeze(1)  ## transform to [Bs, 1, 30 , 30 ]
            real_data.requires_grad_(True)  
            
            real_label = batch[1] ## energy label
            real_label = real_label.unsqueeze(-1)  ## transform to [Bs, 1 ]

            #print("r_label" + str(r_label))
            #end = timer(); print(f'---load real imgs elapsed time: {end-start}')

            start = timer()
            real_data = real_data.to(device)
            real_label = real_label.to(device)

            # train with real data
            disc_real, aux_output = aD(real_data.float(), real_label.float())
            real_label = real_label.view(-1) 
            real_label = (real_label - 10) / 40 
            aux_errD_real = aux_criterion(aux_output, real_label.long() )   ## transform real_labels --> 0 and 1 
            errD_real = aux_errD_real.mean()
            disc_real = disc_real.mean()


            # train with fake data
            disc_fake, aux_output = aD(fake_data, torch.from_numpy(f_label).to(device).float())
            #aux_errD_fake = aux_criterion(aux_output, fake_label)
            #errD_fake = aux_errD_fake.mean()
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            real_labelGP = batch[1] ## energy label
            real_labelGP = real_label.unsqueeze(-1)  ## transform to [Bs, 1 ]
            gradient_penalty = calc_gradient_penalty(aD, real_data, real_labelGP, fake_data)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_acgan = errD_real #+ errD_fake
            (disc_cost + ACGAN_SCALE*disc_acgan).backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                #writer.add_scalar('data/disc_fake', disc_fake, iteration)
                #writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                writer.add_scalar('data/ac_disc_cost', disc_acgan, iteration)
                writer.add_scalar('data/ac_gen_cost', aux_errG, iteration)
                writer.add_scalar('data/wasserstein_distance',  w_dist, iteration)
                #writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                #writer.add_scalar('data/fake_data_mean', fake_data.mean())
                #writer.add_scalar('data/real_data_mean', real_data.mean())
                #if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
                #if iteration %200==199:
                #    body_model = [i for i in aD.children()][0]
                #    layer1 = body_model.conv
                #    xyz = layer1.weight.data.clone()
                #    tensor = xyz.cpu()
                #    tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                #    writer.add_image('D/conv1', tensors, iteration)

            #end = timer(); print(f'---train D elapsed time: {end-start}')
        #---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)
        #if iteration %200==199:
        #   paramsG = aG.named_parameters()
        #   for name, pG in paramsG:
        #       writer.add_histogram('G.' + name, pG.clone().data.cpu().numpy(), iteration)
	    #----------------------Generate images-----------------

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        #print ('iteration: {}, Wasserstein dist: {}'.format(iteration, w_dist.cpu().data.numpy()) )
        if iteration % 1000==999:
        #    val_loader = load_data(VAL_DIR, VAL_CLASS)
        #    dev_disc_costs = []
        #    for _, images in enumerate(val_loader):
        #        imgs = torch.Tensor(images[0])
        #       	imgs = imgs.to(device)
        #        with torch.no_grad():
        #    	    imgs_v = imgs
        #
        #        D, _ = aD(imgs_v)
        #        _dev_disc_cost = -D.mean().cpu().data.numpy()
        #        dev_disc_costs.append(_dev_disc_cost)
        #    lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
        #    lib.plot.flush()	
        #    gen_images = generate_image(aG, fixed_noise)
        #    torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
        #    grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
        #    writer.add_image('images', grid_images, iteration)
            #gen_images = generate_image(iteration, aG, persistant_noise)
            #gen_images = torchvision.utils.make_grid(torch.from_numpy(gen_images), nrow=8, padding=1)
            #writer.add_image('images', gen_images, iteration)
	        #----------------------Save model----------------------
            
            torch.save(aG.state_dict(), 'output/{0}/netG_itrs_{1}.pth'.format(EXP, iteration))
            torch.save(aD.state_dict(), 'output/{0}/netD_itrs_{1}.pth'.format(EXP, iteration))

            
        lib.plot.tick()

train()


