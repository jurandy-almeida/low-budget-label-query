"""Model definition."""

import os
import torch
from torch import nn
import torchvision

import dialnet

class Model(nn.Module):
    def __init__(self, num_class, base_model='resnet18', model_file='', **kwargs):
        super(Model, self).__init__()

        print(("""
Initializing model:
    base model:         {}.
    num_class:          {}.
    model_file:          {}.
        """.format(base_model, num_class, model_file)))

        self._prepare_base_model(base_model, model_file, **kwargs)
        self._prepare_top_layer(num_class)
        self._load_model(num_class, model_file)

    def _prepare_top_layer(self, num_class):

        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))

    def _prepare_base_model(self, base_model, model_file, **kwargs):

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(**kwargs)
        elif 'dialnet' in base_model:
            self.base_model = getattr(dialnet, base_model)(**kwargs)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _load_model(self, num_class, model_file):

        if os.path.isfile(model_file):
            state_dict = torch.load(model_file)
            if 'base_model.fc.bias' in state_dict.keys() and \
               state_dict['base_model.fc.bias'].size(0) != num_class:
                state_dict.pop('base_model.fc.bias')
            if 'base_model.fc.weight' in state_dict.keys() and \
               state_dict['base_model.fc.weight'].size(0) != num_class:
                state_dict.pop('base_model.fc.weight')
            self.load_state_dict(state_dict, strict=False)

    def forward(self, input):
        output = self.base_model(input)
        return output


