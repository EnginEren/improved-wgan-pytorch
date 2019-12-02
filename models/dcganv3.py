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
        self.conv1 = torch.nn.Conv2d(nc+1, ndf*8, kernel_size=5, stride=1, padding=0, bias=False)
        ## layer-normalization
        self.bn1 = torch.nn.LayerNorm([26,26])
        ## convolution
        self.conv2 = torch.nn.Conv2d(ndf*8, ndf*4, kernel_size=4, stride=1, padding=0, bias=False)
        ## layer-normalization
        self.bn2 = torch.nn.LayerNorm([23,23])
        #convolution
        self.conv3 = torch.nn.Conv2d(ndf*4, ndf*2, kernel_size=4, stride=2, padding=0, bias=False)
        ## layer-normalization
        self.bn3 = torch.nn.LayerNorm([10,10])
        #convolution
        self.conv4 = torch.nn.Conv2d(ndf*2, ndf*2, kernel_size=3, stride=1, padding=0, bias=False)
        ## layer-normalization
        self.bn4 = torch.nn.LayerNorm([8,8])
        #convolution
        self.conv5 = torch.nn.Conv2d(ndf*2, ndf, kernel_size=3, stride=1, padding=0, bias=False)

        # Read-out layer : ndf * isize * isize input features, ndf output features 
        self.fc1 = torch.nn.Linear(ndf * 6 * 6, 10)
        self.fc2 = torch.nn.Linear(10,1)
        
    def forward(self, x, energy):
        
        ## conditioning on energy
        energy = F.leaky_relu(self.cond1(energy), 0.2)
        energy = F.leaky_relu(self.cond2(energy), 0.2)
        
        ## reshape into two 2D
        energy = energy.view(-1, 1, self.isize, self.isize)
        
        ## concentration with input : N+1 (Nlayers + 1 cond) x 30 x 30
        x = torch.cat((x, energy), 1)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        #Size changes from (nc+1, 30, 30) to (ndf, 6, 6)

        
        x = x.view(-1, self.ndf * 6 * 6)
        # Size changes from (ndf, 6, 6) to (1, ndf * 6 * 6) 
        #Recall that the -1 infers this dimension from the other given dimension


        # Read-out layer 
        x = F.leaky_relu(self.fc1(x), 0.2)
        output_wgan = self.fc2(x)
        
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
        
        self.cond1 = torch.nn.Linear(self.z+1, 5*5*ngf)
        
        
        ## deconvolution
        self.deconv1 = torch.nn.ConvTranspose2d(ngf, ngf*8, kernel_size=3, stride=3, padding=1, bias=False)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ngf*8)
        ## deconvolution
        self.deconv2 = torch.nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=3, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ngf*4)
        # deconvolution
        self.deconv3 = torch.nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ngf*2)
        
        # deconvolution
        self.deconv4 = torch.nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(ngf)
        # deconvolution
        self.deconv5 = torch.nn.ConvTranspose2d(ngf, 30, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(30)
        # deconvolution
        self.deconv5 = torch.nn.ConvTranspose2d(30, 10, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(10)
        # deconvolution
        self.deconv5 = torch.nn.ConvTranspose2d(10, 1, kernel_size=3, stride=2, padding=1, bias=False)


       
    
        
        
    def forward(self, noise, energy):
         
        layer = []
         ### need to do generated 30 layers, hence the loop!
        for i in range(self.nc):     
            ## conditioning on energy 
            x = F.leaky_relu(self.cond1(torch.cat((energy, noise), 1)), 0.2)
           
            
            x = x.contiguous()
        
            ## change size for deconv2d network. Image is 10x10
            x = x.view(-1,self.ngf,5,5)        

            ## apply series of deconv2d and batch-norm
            x = F.leaky_relu(self.bn1(self.deconv1(x, output_size=[x.size(0), x.size(1) , 10, 10])), 0.2) 
            x = F.leaky_relu(self.bn2(self.deconv2(x, output_size=[x.size(0), x.size(1) , 20, 20])), 0.2)
            x = F.leaky_relu(self.bn3(self.deconv3(x, output_size=[x.size(0), x.size(1) , 30, 30])), 0.2)                         
            
            ##Image is 120x120
            
            ## one standard conv and batch-norm layer 
            x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2)

            layer.append(x)
        
       
        ## concentration of the layers
        x = torch.cat([layer[i] for l in range(self.nc)], 1)

        ## Further apply series of conv and batch norm layers 
        x = F.leaky_relu(self.bn01(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn02(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn03(self.conv3(x)), 0.2)
        x = F.relu(self.conv4(x))

        return x
