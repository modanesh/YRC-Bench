import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info(f"Trainable parameters: {params}")
        self.logger.info(self)


class Network(BaseModel):
    def __init__(self):
        super().__init__()
        self.last_layer_dim = 64  # 32 vs 128
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(2, 16, 5, bias = False, padding = 2)  # in: 1 vs 3, out: 8 vs 32
        self.bn2d1 = nn.BatchNorm2d(16, eps = 1e-04, affine = False)  # num features: 8 vs 32
        self.conv2 = nn.Conv2d(16, 32, 5, bias  =False, padding = 2)
        self.bn2d2 = nn.BatchNorm2d(32, eps = 1e-04, affine = False)
        self.conv3 = nn.Conv2d(32, 64, 5, bias = False, padding = 2)
        self.bn2d3 = nn.BatchNorm2d(64, eps = 1e-04, affine = False)
        self.fc1 = nn.Linear(64 * 4 * 4, self.last_layer_dim, bias = False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class AutoEncoder(BaseModel):
    def __init__(self):
        super().__init__()
        self.last_layer_dim = 64  # must match network above
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        self.conv1 = nn.Conv2d(2, 16, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.conv1.weight, gain = nn.init.calculate_gain("leaky_relu"))
        self.bn2d1 = nn.BatchNorm2d(16, eps = 1e-04, affine = False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.conv2.weight, gain = nn.init.calculate_gain("leaky_relu"))
        self.bn2d2 = nn.BatchNorm2d(32, eps = 1e-04, affine = False)
        self.conv3 = nn.Conv2d(32, 64, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.conv3.weight, gain = nn.init.calculate_gain("leaky_relu"))
        self.bn2d3 = nn.BatchNorm2d(64, eps = 1e-04, affine = False)
        self.fc1 = nn.Linear(64 * 4 * 4, self.last_layer_dim, bias = False)
        self.bn1d = nn.BatchNorm1d(self.last_layer_dim, eps = 1e-04, affine = False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.last_layer_dim / (4 * 4)), 64, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain = nn.init.calculate_gain("leaky_relu"))
        self.bn2d4 = nn.BatchNorm2d(64, eps = 1e-04, affine = False)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain = nn.init.calculate_gain("leaky_relu"))
        self.bn2d5 = nn.BatchNorm2d(32, eps = 1e-04, affine = False)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain = nn.init.calculate_gain("leaky_relu"))
        self.bn2d6 = nn.BatchNorm2d(16, eps = 1e-04, affine = False)
        self.deconv4 = nn.ConvTranspose2d(16, 2, 5, bias = False, padding = 2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain = nn.init.calculate_gain("leaky_relu"))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))

        x = x.view(x.size(0), int(self.last_layer_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)

        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)

        x = torch.sigmoid(x)
        return x

    def summary(self):
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info(f"Trainable parameters: {params}")
        self.logger.info(self)
