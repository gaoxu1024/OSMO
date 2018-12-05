import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.model.resnet import ResNet
from PIL import Image

# Parameters
learning_rate = 1e-3
input_dim = 3  # The dimension of the input image
EPOCH = 300
batch_size = 20
seq_len = 20  # The sequence length of the training sequences.
img_h, img_w = 224, 224
seq_times = 2  # How many times does one LONG sequence appear in the training set
test_seq_times = 1  # How many times does one LONG sequence appear in the testing set
train_ds = ['iLIDS-VID', 'PRID2011', 'MARS']
# train_ds = ['iLIDS-VID']
test_ds = ['iLIDS-VID', 'PRID2011', 'MARS']
# test_ds = ['iLIDS-VID']
# test_ds = ['CrowdedCrossing', 'NightCrossing', 'MetroOut', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']

# resnet = models.vgg16_bn(pretrained=True).cuda()
resnet = ResNet(18).cuda()
resnet_dict = resnet.state_dict()

pretrained_resnet = torch.load('utils/model/model_best.pth.tar')  # ResNet18
pretrained_dict = {k: v for k, v in pretrained_resnet['state_dict'].items() if k in resnet_dict}
resnet_dict.update(pretrained_dict)
resnet.load_state_dict(resnet_dict)

resnet.base.fc = nn.Linear(512, 128)

# Online Appearance Model (CNN+RNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CNN Part
        self.feature_map = nn.Sequential(
            *list(resnet.base.children())[:-2]
        )

        self.feature_pool = resnet.base.avgpool
        self.feature_fc = resnet.base.fc

        # self.temporal_fc = nn.Linear(seq_len*256, 256)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            # nn.Dropout(0.5),
            nn.Tanh(),
        )

        self.metric_subnet_conv = nn.Conv2d(512, 128, 1)

        self.metric_subnet_fc = nn.Sequential(
            nn.Linear(128*7*7, 1),
            nn.Sigmoid(),
        )

        self.metric_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # RNN Part
        self.hidden_dim = 128
        # self.rnn = nn.LSTM(input_size=256, hidden_size=self.hidden_dim, num_layers=1)
        self.rnncell = nn.LSTMCell(input_size=128, hidden_size=self.hidden_dim)

    def forward(self, input):
        x, y = input[:, :-1], input[:, -1]

        y_map = self.feature_map(y)
        y_out = self.feature_fc(self.feature_pool(y_map).view(y_map.shape[0], -1))
        y_out = self.fc(y_out)

        h = Variable(torch.zeros(x.shape[0], self.hidden_dim)).cuda()  # Initialize the hidden layer h
        c = Variable(torch.zeros(x.shape[0], self.hidden_dim)).cuda()  # Initialize the hidden layer c
        for frame in range(x.shape[1]):
            # import pdb
            # pdb.set_trace()
            xt_map = self.feature_map(x[:, frame])
            # import pdb
            # pdb.set_trace()
            xt_input = self.feature_fc(self.feature_pool(xt_map).view(xt_map.shape[0], -1))

            diff_map = torch.abs(xt_map - y_map)
            diff_map = self.metric_subnet_conv(diff_map)
            diff_map = diff_map.view(diff_map.shape[0], -1)
            prob = self.metric_subnet_fc(diff_map)
            current_input = prob * xt_input + (1 - prob) * h  # Balance the current frame and previous frames
            h, c = self.rnncell(current_input, (h, c))

        x_out = h  # Output of the RNN network
         # x_out, h = self.rnn(rnn_input, None)
        # x_out = x_out.permute(1, 0, 2).contiguous()  # Change the first and the second dimension of x_out.
        # x_out = x_out.view(x_out.size(0), -1)
        # x_out = self.temporal_fc(x_out)
        # x_out = torch.mean(x_out, 0)  # Temporal Average Pooling
        # x_out = self.fc(x_out[seq_len-1])

        diff_final = torch.abs(x_out - y_out)
        prob = self.metric_fc(diff_final)
        return prob

