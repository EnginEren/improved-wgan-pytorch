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

    def __init__(self, isize, nc, ndf, nclass):
        super(DCGAN_D, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.nclass = nclass


        ### convolution
        self.conv1 = torch.nn.Conv2d(nc, ndf*8, kernel_size=4, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ndf*8)
        ## convolution
        self.conv2 = torch.nn.Conv2d(ndf*8, ndf*4, kernel_size=4, stride=1, padding=1, bias=False)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ndf*4)
        #convolution
        self.conv3 = torch.nn.Conv2d(ndf*4, ndf*2, kernel_size=4, stride=1, padding=1, bias=False)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ndf*2)
        #convolution
        self.conv4 = torch.nn.Conv2d(ndf*2, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn4 = torch.nn.BatchNorm2d(ndf)
        #convolution
        self.conv5 = torch.nn.Conv2d(ndf, 1, kernel_size=2, stride=2, padding=1, bias=False)
        # Read-out layer : 4 * 4  input features, ndf output features 
        self.fc = torch.nn.Linear((4 * 4)+1, 1)
        self.fc2 = torch.nn.Linear((4 * 4)+1, self.nclass)


        
    def forward(self, x, energy):
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True) # 15 x 15
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True) # 14 x 14
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True) # 13 x 13
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, inplace=True)  # 6x6
        x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)  # 4x4

        #After series of convlutions --> size changes from (nc, 30, 30) to (1, 4, 4)

        
        x = x.view(-1, 4 * 4) 
        x = torch.cat((x, energy), 1)
        
        # Size changes from (ndf, 30, 30) to (1, (4 * 4) + 1) 
        #Recall that the -1 infers this dimension from the other given dimension


        # Read-out layers 
        output_wgan = self.fc(x)
        output_congan = self.fc2(x)
        
        output_wgan = output_wgan.view(-1) ### flattens

        return output_wgan, output_congan




class DCGAN_G(nn.Module):
    """ 
    generator component of WGAN

    """
    def __init__(self, nc, ngf, z):
        super(DCGAN_G, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        
    
        
        ## deconvolution
        self.deconv1 = torch.nn.ConvTranspose2d(z, ngf*8, kernel_size=4, stride=1, padding=0, bias=False)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ngf*8)
        ## deconvolution
        self.deconv2 = torch.nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ngf*4)
        # deconvolution
        self.deconv3 = torch.nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ngf*2)
        # deconvolution
        self.deconv4 = torch.nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn4 = torch.nn.BatchNorm2d(ngf)
        # deconvolution
        self.deconv5 = torch.nn.ConvTranspose2d(ngf, 1, kernel_size=1, stride=1, padding=1, bias=False)


        
        
    def forward(self, z):
        
        z = z.view(-1,self.z,1,1)
        
        layer = []
        ## need to do generate N layers, hence the loop!
        for i in range(self.nc):     
    

            ## apply series of deconv2d and batch-norm
            x = F.leaky_relu(self.bn1(self.deconv1(z)), 0.2, inplace=True)  # 4 x 4 
            x = F.leaky_relu(self.bn2(self.deconv2(x)), 0.2, inplace=True)  # 8 x 8
            x = F.leaky_relu(self.bn3(self.deconv3(x)), 0.2, inplace=True)  # 16 x 16 
            x = F.leaky_relu(self.bn4(self.deconv4(x)), 0.2, inplace=True)  # 32 x 32
            x = F.relu(self.deconv5(x))                                     # 30 x 30

            ##Image is 30x30 now
            
           

            layer.append(x)
        
       
        ## concentration of the layers
        x = torch.cat([layer[i] for l in range(self.nc)], 1)


        return x
