import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dialnet

from model import Model
from transforms import Map
from datasets import CIFAR9, STL9
from classifier import DLClassifier

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet")
    and callable(models.__dict__[name]))
model_names = model_names + ['dialnet']

dataset_names = ['CIFAR9', 'STL9', 'MNIST', 'SVHN', 'USPS']

def main():
    parser = argparse.ArgumentParser(description='PyTorch CoDIAL + UNFOLD')
    parser.add_argument('--data', '-d', metavar='DATA', required=True,
                        choices=dataset_names,
                        help='image datasets: ' + ' | '.join(dataset_names))
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dialnet',
                        choices=model_names,
                        help='model architectures: ' + ' | '.join(model_names) +
                        ' (default: dialnet)')
    parser.add_argument('-w', '--weights', required=True, type=str, metavar='PATH',
                        help='path to the weights of the trained model')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true',
                        help='test on grayscale images')
    args = parser.parse_args()

    cudnn.benchmark = True

    params = dict()
    params['num_workers'] = args.workers
    params['batch_size'] = args.batch_size
    params['print_freq'] = args.print_freq
    params['half'] = args.half

    print('Testing arguments:')
    for k, v in params.items():
        print('\t{}: {}'.format(k, v))

    test_transform = []

    if 'dialnet' in args.arch:
       resize = transforms.Resize(32)
       test_transform.append(resize)

    print("=> preparing testing data '{}'".format(args.data))

    test_transform.append(transforms.ToTensor())

    if args.data == 'CIFAR9':
        normalize = transforms.Normalize(mean=[0.424, 0.415, 0.384],
                                         std=[0.283, 0.278, 0.284])
        test_transform.append(normalize)
        test_data = CIFAR9(root='./data', train=False, transform=transforms.Compose(test_transform), download=True)
    elif args.data == 'STL9':
        normalize = transforms.Normalize(mean=[0.447, 0.440, 0.407],
                                         std=[0.260, 0.257, 0.271])
        test_transform.append(normalize)
        test_data = STL9(root='./data', split='test', transform=transforms.Compose(test_transform), download=True)
    elif args.data == 'MNIST':
        # to make MNIST RGB instead of grayscale
        gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        normalize = transforms.Normalize(mean=[0.5] if args.grayscale else [0.5, 0.5, 0.5],
                                         std=[0.5] if args.grayscale else [0.5, 0.5, 0.5])
        test_transform.extend([normalize] if args.grayscale else [gray2rgb, normalize])
        test_data = datasets.__dict__[args.data](root='./data', train=False, transform=transforms.Compose(test_transform), download=True)
    elif args.data == 'SVHN':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        test_transform.append(normalize)
        test_data = datasets.__dict__[args.data](root='./data/SVHN', split='test', transform=transforms.Compose(test_transform), download=True)
    elif args.data == 'USPS':
        normalize = transforms.Normalize(mean=[0.5],
                                                std=[0.5])
        test_transform.append(normalize)
        test_data = datasets.__dict__[args.data](root='./data', train=False, transform=transforms.Compose(test_transform), download=True)

    if args.data in ['CIFAR9', 'STL9']:
        num_classes = 9
        arch = 'cifar9-stl9'
    elif args.data == 'SVHN' or (args.data == 'MNIST' and not args.grayscale):
        num_classes = 10
        arch = 'mnist-svhn'
    elif args.data == 'USPS' or (args.data == 'MNIST' and     args.grayscale):
        num_classes = 10
        arch = 'mnist-usps'
    else:
        raise NotImplementedError

    print("=> loading model '{}'".format(args.arch))

    model_params = dict()
    if 'resnet' in args.arch:
        model_params['pretrained'] = True
    elif 'dialnet' in args.arch:
        model_params['training_mode'] = 'single'
        model_params['arch'] = arch
    model = Model(num_classes, base_model=args.arch, model_file=args.weights, **model_params)

    criterion = nn.CrossEntropyLoss()

    classifier = DLClassifier(model, criterion, params)

    with torch.no_grad():
        classifier.predict(test_data, accuracy=True)

if __name__ == '__main__':
    main()
