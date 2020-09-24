import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import torch_neuron
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace = False)
        relu1_1 = h
        h = F.relu(self.conv1_2(h), inplace = False)
        h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace = False)
        relu2_1 = h
        h = F.relu(self.conv2_2(h), inplace = False)
        h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace = False)
        relu3_1 = h
        h = F.relu(self.conv3_2(h), inplace = False)
        relu3_2 = h
        h = F.relu(self.conv3_3(h), inplace = False)
        h = F.relu(self.conv3_4(h), inplace = False)
        h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.conv4_1(h)
        conv4_1 = h
        h = F.relu(h, inplace = False)
        relu4_1 = h

        return [relu1_1, relu2_1, relu3_1, relu4_1, relu3_2, conv4_1]

if __name__ == "__main__":
    model = Vgg19()
    model.load_state_dict(torch.load('vgg_small_statedict.pth'))
    model.eval()

    image = torch.zeros([1, 3, 512, 512], dtype=torch.float32)
    compiler_args=['-O2', '--static-weights', '--num-neuroncores', '4']

    tic = time.clock()
    model_neuron = torch.neuron.trace(model, example_inputs=[image], compiler_args=compiler_args)
    toc = time.clock()
    print("\nElapsed Time = {} seconds".format(toc-tic))

    ## Export to saved model
    model_neuron.save("vgg_small_neuron_4.pt")
