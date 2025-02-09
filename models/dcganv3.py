import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

class DCGAN_D(nn.Module):
    """ 
    discriminator component of WGAN
    """

    def __init__(self, isize, nc, ndf):
        super(DCGAN_D, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        ## linear layers
        self.cond1 = torch.nn.Linear(1, 50)
        self.cond2 = torch.nn.Linear(50, isize*isize)
        
        ### convolution
        self.conv1 = torch.nn.Conv2d(nc+1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        ## layer-normalization
        self.bn1 = torch.nn.LayerNorm([14,14])
        ## convolution
        self.conv2 = torch.nn.Conv2d(ndf, 10, kernel_size=3, stride=2, padding=0, bias=False)
        ## layer-normalization
        self.bn2 = torch.nn.LayerNorm([6,6])
        #convolution
        self.conv3 = torch.nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=0, bias=False)

        # Read-out layer : ndf * isize * isize input features, ndf output features 
        self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        
    def forward(self, x, energy):
        
        ## conditioning on energy
        energy = F.leaky_relu(self.cond1(energy), 0.2)
        energy = self.cond2(energy)
        
        ## reshape into two 2D
        energy = energy.view(-1, 1, self.isize, self.isize)
        
        ## concentration with input : N+1 (Nlayers + 1 cond) x 30 x 30
        x = torch.cat((x, energy), 1)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        #Grand total --> size changes from (nc+1, 30, 30) to (5, 4, 4)

        
        x = x.view(-1, 5 * 4 * 4)
        # Size changes from (5, 4, 4) to (1, 5 * 4 * 4) 
        #Recall that the -1 infers this dimension from the other given dimension


        # Read-out layer 
        output_wgan = self.fc(x)
        
        output_wgan = output_wgan.view(-1) ### flattens

        return output_wgan


class DCGAN_G(nn.Module):
    """ 
    generator component of WGAN
    """
    def __init__(self, nc, ngf, z):
        super(DCGAN_G, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        
        self.cond1 = torch.nn.Linear(self.z+1, 200)
        self.cond2 = torch.nn.Linear(200, 5*5*ngf)
        
        ## deconvolution  
        self.deconv1 = torch.nn.ConvTranspose2d(ngf, ngf*8, kernel_size=2, stride=2, padding=1, bias=False) 
        
        
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ngf*8)
        
        ## deconvolution 8x8 -- > 14x14
        self.deconv2 = torch.nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=2, stride=2, padding=1, bias=False)
        
        
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ngf*4)
        
        # deconvolution 14x14 -- > 26x26
        self.deconv3 = torch.nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=2, stride=2, padding=1, bias=False)
       
        
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ngf*2)
        
        # deconvolution 26x26 -- > 27x27
        self.deconv4 = torch.nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=1, padding=1, bias=False)
        
        self.bn4 = torch.nn.BatchNorm2d(ngf)
        
        # deconvolution 27x27 -- > 30x30
        self.deconv5 = torch.nn.ConvTranspose2d(ngf, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
  
    
        
        
    def forward(self, noise, energy):
         
        layer = []
         ### need to do generated 30 layers, hence the loop!
        for i in range(self.nc):     
            ## conditioning on energy 
            x = F.leaky_relu(self.cond1(torch.cat((energy, noise), 1)), 0.2)
            x = self.cond2(x)
            
            #x = x.contiguous()
        
            ## change size for deconv2d network. Image is 10x10
            x = x.view(-1,self.ngf,5,5)        

           
            ## apply series of deconv2d and batch-norm     
            x = self.deconv1(x)
            x = F.leaky_relu(self.bn1(x), 0.2)
            x = self.deconv2(x)
            x = F.leaky_relu(self.bn2(x), 0.2)
            x = self.deconv3(x)
            x = F.leaky_relu(self.bn3(x), 0.2)
            x = self.deconv4(x)
            x = F.leaky_relu(self.bn4(x), 0.2)
            x = F.relu(self.deconv5(x))
            
           

            layer.append(x)
        
        ## concentration of the layers
        x = torch.cat([layer[i] for l in range(self.nc)], 1)

       

        return x