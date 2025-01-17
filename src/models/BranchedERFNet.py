"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet


class BranchedERFNet(nn.Module):
    '''shared encoder + 2 branched decoders'''
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        self.num_classes = num_classes
        print('Creating branched erfnet with {} classes'.format(num_classes))
        # num_classes=[3, 1], 3 for instance-branch (1 sigma) & 1 for seed branch
        # num_classes=[4, 1], 4 for instance-branch (2 sigma) & 1 for seed branch
        
        # shared encoder
        if (encoder is None):
            self.encoder = erfnet.Encoder(sum(self.num_classes))  # Encoder(3+1)
        else:
            self.encoder = encoder
        
        # decoder for 2 branches (instance & seed)
        # Decoder(3) for instance branch & Decoder(1) for seed branch
        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        if sum(self.num_classes) == 4:
            n_sigma = 1  # 1 sigma for circular margin
        else:
            n_sigma = 2  # 2 sigma for elliptical margin

        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:  # not used in this code
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # (N, 128, h/8, w/8)
        
        # concat (N, 3, h, w) & (N, 1, h, w)
        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)
