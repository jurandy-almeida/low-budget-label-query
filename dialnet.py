import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import EntropyLoss, ConsistencyLoss

from utils import rsetattr

def dialnet(arch='cifar9-stl9', **kwargs):
    model = DIALNet(arch, **kwargs)
    return model


class DIALEntropyLoss(nn.Module):

    __constants__ = ['entropy_loss_weight']

    def __init__(self, entropy_loss_weight=0.1):
        super(DIALEntropyLoss, self).__init__()
        self.entropy_loss_weight = entropy_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.en_loss = EntropyLoss()

    def forward(self, x, y):
        if self.training:
            xs, xt = torch.split(x, split_size_or_sections=x.size(0) // 2, dim=0)
            ce_loss = self.ce_loss(xs, y)
            en_loss = self.en_loss(xt)
            return ce_loss + self.entropy_loss_weight * en_loss
        else:
            return self.ce_loss(x, y)

    def extra_repr(self):
        s = ('entropy_loss_weight={entropy_loss_weight}')
        return s.format(**self.__dict__)


class DIALConsistencyLoss(nn.Module):

    __constants__ = ['consistency_loss_weight']

    def __init__(self, consistency_loss_weight=0.1):
        super(DIALConsistencyLoss, self).__init__()
        self.consistency_loss_weight = consistency_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.en_loss = EntropyLoss()
        self.co_loss = ConsistencyLoss(reduction='batchmean')

    def forward(self, x, y):
        if self.training:
            xs, xt, xp = torch.split(x, split_size_or_sections=x.size(0) // 3, dim=0)
            ce_loss = self.ce_loss(xs, y)
            en_loss = self.en_loss(xt)
            co_loss = self.co_loss(xt, xp)
            return ce_loss + self.consistency_loss_weight * (en_loss + co_loss)
        else:
            return self.ce_loss(x, y)

    def extra_repr(self):
        s = ('consistency_loss_weight={consistency_loss_weight}')
        return s.format(**self.__dict__)


class DIALNet(nn.Module):

    cfg_mnist_usps = [32, 'M', 48, 'M', 'F', 'F']
    cfg_mnist_svhn = [64, 'M', 64, 'M', 128, 'F', 'F']
    cfg_cifar9_stl9 = [128, 128, 128, 'M', 256, 256, 256, 'M', 512, 'P', 'P']

    def __init__(self, arch='cifar9-stl9', num_classes=10, training_mode='dual'):
        super(DIALNet, self).__init__()
        self.features, out_size, out_channels = self.make_layers(arch, training_mode)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if out_size > 1 else nn.Identity()
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def make_layers(self, arch, training_mode):
        layers = []
        in_size = 32
        in_channels = 1 if arch == 'mnist-usps' else 3
        if arch == 'cifar9-stl9':
            for v in self.cfg_cifar9_stl9:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)]
                    in_size = (in_size - 2) // 2 + 1
                elif v == 'P':
                    layers += [BasicConv2d(training_mode, in_channels, in_channels // 2, kernel_size=1, padding=0)]
                    in_channels = in_channels // 2
                else:
                    layers += [BasicConv2d(training_mode, in_channels, v, kernel_size=3, padding=1)]
                    in_channels = v
        elif arch == 'mnist-svhn':
            reduction = 2.0
            for v in self.cfg_mnist_svhn:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
                    in_size = (in_size - 3) // 2 + 1
                elif v == 'F':
                    layers += [nn.Flatten()] if in_size > 1 else []
                    in_features = in_size * in_size * in_channels
                    out_features = 1024 * int((in_features // 1024) / reduction)
                    layers += [BasicLinear(training_mode, in_features, out_features)]
                    reduction = (reduction + 1.0) / reduction
                    in_channels = out_features
                    in_size = 1
                else:
                    layers += [BasicConv2d(training_mode, in_channels, v, kernel_size=5, padding=2)]
                    in_channels = v
        elif arch == 'mnist-usps':
            for v in self.cfg_mnist_usps:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    in_size = (in_size - 2) // 2 + 1
                elif v == 'F':
                    if in_size > 1:
                        layers += [nn.Flatten()]
                    in_features = in_size * in_size * in_channels
                    layers += [BasicLinear(training_mode, in_features, 100)]
                    in_channels = 100
                    in_size = 1
                else:
                    layers += [BasicConv2d(training_mode, in_channels, v, kernel_size=5, padding=2)]
                    in_channels = v
        return nn.Sequential(*layers), in_size, in_channels


class BasicConv2d(nn.Module):

    __constants__ = ['training_mode']

    def __init__(self, training_mode, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if training_mode in ['dual', 'triple']:
            self.bns = nn.BatchNorm2d(out_channels, affine=False)
            if training_mode == 'triple':
                self.bnp = nn.BatchNorm2d(out_channels, affine=False)
        self.bnt = nn.BatchNorm2d(out_channels, affine=False)
        self.gamma = nn.Parameter(torch.Tensor(out_channels))
        self.beta = nn.Parameter(torch.Tensor(out_channels))
        self.training_mode = training_mode
        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.training and self.training_mode in ['dual', 'triple']:
            if self.training_mode == 'dual':
                xs, xt = torch.split(x, split_size_or_sections=x.size(0) // 2, dim=0)
                xs = self.bns(xs)
                xt = self.bnt(xt)
                x = torch.cat((xs, xt), dim=0)
            elif self.training_mode == 'triple':
                xs, xt, xp = torch.split(x, split_size_or_sections=x.size(0) // 3, dim=0)
                xs = self.bns(xs)
                xt = self.bnt(xt)
                xp = self.bnp(xp)
                x = torch.cat((xs, xt, xp), dim=0)
        else:
            x = self.bnt(x)
        x = x * self.gamma[:, None, None] + self.beta[:, None, None]
        return F.relu(x, inplace=True)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def extra_repr(self):
        s = ('training_mode={training_mode}')
        return s.format(**self.__dict__)


class BasicLinear(nn.Module):

    __constants__ = ['training_mode']

    def __init__(self, training_mode, in_features, out_features, **kwargs):
        super(BasicLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features, **kwargs)
        self.dropout = nn.Dropout(0.5)
        if training_mode in ['dual', 'triple']:
            self.bns = nn.BatchNorm1d(out_features, affine=False)
            if training_mode == 'triple':
                self.bnp = nn.BatchNorm1d(out_features, affine=False)
        self.bnt = nn.BatchNorm1d(out_features, affine=False)
        self.gamma = nn.Parameter(torch.Tensor(out_features))
        self.beta = nn.Parameter(torch.Tensor(out_features))
        self.training_mode = training_mode
        self.reset_parameters()

    def forward(self, x):
        x = self.fc(x)
        if self.training and self.training_mode in ['dual', 'triple']:
            if self.training_mode == 'dual':
                xs, xt = torch.split(x, split_size_or_sections=x.size(0) // 2, dim=0)
                xs = self.bns(xs)
                xt = self.bnt(xt)
                x = torch.cat((xs, xt), dim=0)
            elif self.training_mode == 'triple':
                xs, xt, xp = torch.split(x, split_size_or_sections=x.size(0) // 3, dim=0)
                xs = self.bns(xs)
                xt = self.bnt(xt)
                xp = self.bnp(xp)
                x = torch.cat((xs, xt, xp), dim=0)
        else:
            x = self.bnt(x)
        x = F.relu(x * self.gamma + self.beta, inplace=True)
        return self.dropout(x)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def extra_repr(self):
        s = ('training_mode={training_mode}')
        return s.format(**self.__dict__)


class DIALBatchNorm2d(nn.Module):

    def __init__(self, training_mode, num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        super(DIALBatchNorm2d, self).__init__()
        if training_mode in ['dual', 'triple']:
            self.bns = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
            if training_mode == 'triple':
                self.bnp = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.bnt = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.gamma = nn.Parameter(torch.Tensor(num_features))
        self.beta = nn.Parameter(torch.Tensor(num_features))
        self.training_mode = training_mode
        self.reset_parameters()

    def forward(self, x):
        if self.training and self.training_mode in ['dual', 'triple']:
            if self.training_mode == 'dual':
                xs, xt = torch.split(x, split_size_or_sections=x.size(0) // 2, dim=0)
                xs = self.bns(xs)
                xt = self.bnt(xt)
                x = torch.cat((xs, xt), dim=0)
            elif self.training_mode == 'triple':
                xs, xt, xp = torch.split(x, split_size_or_sections=x.size(0) // 3, dim=0)
                xs = self.bns(xs)
                xt = self.bnt(xt)
                xp = self.bnp(xp)
                x = torch.cat((xs, xt, xp), dim=0)
        else:
            x = self.bnt(x)
        x = x * self.gamma[:, None, None] + self.beta[:, None, None]
        return x

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def extra_repr(self):
        s = ('training_mode={training_mode}')
        return s.format(**self.__dict__)


class DIAL(nn.Module):

    def __init__(self, base_model, training_mode='dual', **kwargs):
        super(DIAL, self).__init__()
        if not isinstance(base_model, nn.Module):
             raise RuntimeError('A model must be provided')

        self.base_model = base_model

        dial = dict()
        for name, layer in self.base_model.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                dialbn = DIALBatchNorm2d(training_mode, layer.num_features, **kwargs)
                state_dict = layer.state_dict()
                dialbn.bnt.load_state_dict(state_dict, strict=False)
                if training_mode in ['dual', 'triple']:
                    dialbn.bns.load_state_dict(state_dict, strict=False)
                    if training_mode == 'triple':
                        dialbn.bnp.load_state_dict(state_dict, strict=False)
                if 'weight' in state_dict.keys():
                    dialbn.gamma.data.copy_(state_dict['weight'].data)
                if 'bias' in state_dict.keys():
                    dialbn.beta.data.copy_(state_dict['bias'].data)
                dial[name] = dialbn

        for key, value in dial.items():
            rsetattr(self.base_model, key, value)

    def forward(self, input):
        output = self.base_model(input)
        return output

