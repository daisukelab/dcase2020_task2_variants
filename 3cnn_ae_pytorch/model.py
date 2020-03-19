import torch
from torch import nn
import torch.nn.functional as F
import torchsummary
import torch
import pytorch_lightning as pl
import random


class CNNAutoEncoder(nn.Module):
    """
    Thanks to http://dl-kento.hatenablog.com/entry/2018/02/22/200811#Convolutional-AutoEncoder
    """

    def  __init__(self, z_dim=40):
        super().__init__()

        # define the network
        # encoder
        self.conv1 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                              nn.Conv2d(1, 32, kernel_size=5, stride=2),
                              nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),
                              nn.Conv2d(32, 64, kernel_size=5, stride=2),
                              nn.ReLU(), nn.Dropout(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                              nn.ReLU(), nn.Dropout(0.3))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
                              nn.ReLU(), nn.Dropout(0.3))
        self.fc1 = nn.Conv2d(256, z_dim, kernel_size=3)

        # decoder
        self.fc2 = nn.Sequential(nn.ConvTranspose2d(z_dim, 256, kernel_size=3),
                            nn.ReLU(), nn.Dropout(0.3))
        self.conv4d = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
                               nn.ReLU(), nn.Dropout(0.3))
        self.conv3d = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
                               nn.ReLU(), nn.Dropout(0.2))
        self.conv2d = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
                               nn.ReLU())
        self.conv1d = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)

    def forward(self, x):
        encoded = self.fc1(self.conv4(self.conv3(self.conv2(self.conv1(x)))))

        decoded = self.fc2(encoded)
        decoded = self.conv4d(decoded)
        decoded = self.conv3d(decoded)
        decoded = self.conv2d(decoded)[:,:,1:-1,1:-1]
        decoded = self.conv1d(decoded)[:,:,0:-1,0:-1]
        decoded = nn.Sigmoid()(decoded)

        return decoded
