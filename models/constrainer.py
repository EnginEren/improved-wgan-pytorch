import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import models.HDF5Dataset as H
from torch.utils import data

class energyRegressor(nn.Module):
    """ 
    Energy regressor of WGAN. 

    """

    def __init__(self, nc):
        super(energyRegressor, self).__init__()
        self.nc = nc
        self.conv1 = torch.nn.Conv3d(1, 1, kernel_size=(self.nc,3,3), stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(1)
        self.conv2 = torch.nn.Conv3d(1, 16, kernel_size=(self.nc,6,3), stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(16)
        self.conv3 = torch.nn.Conv3d(16, 32, kernel_size=(self.nc,6,3), stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm3d(32)
        self.conv4 = torch.nn.Conv3d(32, 32, kernel_size=(self.nc,6,3), stride=1, padding=0, bias=False)
        self.bn4 = torch.nn.BatchNorm3d(32)
        self.conv5 = torch.nn.Conv3d(32, 64, kernel_size=(self.nc,3,3), stride=1, padding=0, bias=False)
        self.bn5 = torch.nn.BatchNorm3d(64)
        self.conv6 = torch.nn.Conv3d(64, 64, kernel_size=(self.nc,3,3), stride=1, padding=0, bias=False)
        
        
        self.fc1 = torch.nn.Linear(64 * 18 * 9 * self.nc, 50)
        self.fc2 = torch.nn.Linear(50, 1)
        
    def forward(self, x):
        #input shape :  [nc, 30, 30]
        ## reshape the input: expand one dim
        x = x.unsqueeze(1)
        
        ## image [nc, 30, 30]
        ### convolution and batch normalisation
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        
        ## shape [nc, 9, 18]
        
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        ## pass to FC layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        
        #print (x.shape)
        
        return x
    