import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dialnet

from transforms import GaussianBlur, RandomAffine
from datasets import Perturbate, Sample, Subset
from sampler import Random, TopRank, Uniform
from scorer import Entropy, Consistency
from cluster import DLCluster
from classifier import DLClassifier
from utils import Range, PathType

from utils import get_file_with_parents

import tblog

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(models.__dict__[name]))

training_modes = ['single', 'dual', 'triple']

dataset_names = ['amazon', 'dslr', 'webcam']

sampler_names = ['random', 'toprank', 'uniform']

scorer_names = ['entropy', 'consistency']


def main():
    parser = argparse.ArgumentParser(description='PyTorch CoDIAL + UNFOLD Office-31')
    parser.add_argument('--root', '-r', type=PathType(exists=True, type='dir'),
                        metavar='PATH', default='data/Office31',
                        help='path to the OfficeHome dataset')
    parser.add_argument('--source', '-s', metavar='DATA', default='amazon',
                        choices=dataset_names,
                        help='image datasets: ' + ' | '.join(dataset_names) +
                        ' (default: amazon)')
    parser.add_argument('--target', '-t', metavar='DATA', default='webcam',
                        choices=dataset_names,
                        help='image datasets: ' + ' | '.join(dataset_names) +
                        ' (default: webcam)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architectures: ' + ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--mode', '-m', metavar='MODE', default='dual',
                        choices=training_modes,
                        help='training modes: ' + ' | '.join(training_modes) +
                        ' (default: dual)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run (default: 60)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 20)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--lr-steps', default=[54], type=int, nargs="+",
                        metavar='N', help='epochs to decay learning rate.')
    parser.add_argument('--gamma', default=0.1, type=float, metavar='GAMMA',
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-c', '--clustering', dest='clustering', action='store_true',
                        help='perform clustering on training set')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-every', dest='save_every',
                        help='saves checkpoints at every specified number of epochs (default: 10)',
                        type=int, default=10)
    parser.add_argument('--sampler',  metavar='FUNC', default='',
                        choices=sampler_names,
                        help='selection criteria: ' + ' | '.join(sampler_names) +
                        ' (default: none)')
    parser.add_argument('--sample-ratio', '--sr', default=0.1, type=float,
                        choices=Range(0.0, 1.0),
                        metavar='SR', help='sample ratio (default: 0.1)')
    parser.add_argument('--scorer',  metavar='FUNC', default='',
                        choices=scorer_names,
                        help='selection criteria: ' + ' | '.join(scorer_names) +
                        ' (default: none)')
    parser.add_argument('--subset', default=None, type=float, nargs="+",
                        metavar='N', help='extracts a stratified random sample (default: none)')
    parser.add_argument('--useall', dest='useall', action='store_true',
                        help='use all data for training')
    args = parser.parse_args()

    cudnn.benchmark = True

    params = dict()
    params['num_workers'] = args.workers
    params['epochs'] = args.epochs
    params['start_epoch'] = args.start_epoch
    params['batch_size'] = args.batch_size
    params['lr'] = args.lr
    params['momentum'] = args.momentum
    params['weight_decay'] = args.weight_decay
    params['lr_steps'] = args.lr_steps
    params['gamma'] = args.gamma
    params['print_freq'] = args.print_freq
    params['resume'] = args.resume
    params['pretrained'] = args.pretrained
    params['half'] = args.half
    params['save_every'] = args.save_every

    print('Training arguments:')
    for k, v in params.items():
        print('\t{}: {}'.format(k, v))

    cluster_prefix = os.path.basename(args.root).lower() + '_' + \
                     args.source.lower() + '_' + \
                     args.target.lower() + '_' + \
                     args.arch.lower() + '_' + \
                     args.mode.lower()
    cluster_dir = os.path.join('clusters', cluster_prefix)

    if not os.path.isdir(cluster_dir):
        os.makedirs(cluster_dir)
        if tblog.is_enabled:
            tblog.logdir = cluster_dir

    source_train_transform = []
    target_train_transform = []
    target_val_transform = []

    resize_and_crop = transforms.Compose([ transforms.Resize(256), transforms.RandomCrop(224) ])
    source_train_transform.append(resize_and_crop)
    target_train_transform.append(resize_and_crop)
    target_val_transform.append(resize_and_crop)

    # optionally train on source
    print("=> preparing source data '{}'".format(args.source))

    source_train_transform.append(transforms.ToTensor())

    source_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    source_train_transform.append(source_normalize)
    source_train_data = datasets.ImageFolder(root=os.path.join(args.root, args.source), transform=transforms.Compose(source_train_transform))

    print("=> preparing target data '{}'".format(args.target))

    target_train_transform.append(transforms.ToTensor())

    target_val_transform.append(transforms.ToTensor())

    target_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    target_train_transform.append(target_normalize)
    target_val_transform.append(target_normalize)
    target_train_data = Subset(datasets.ImageFolder(root=os.path.join(args.root, args.target)), proportions=args.subset)
    target_val_data = datasets.ImageFolder(root=os.path.join(args.root, args.target), transform=transforms.Compose(target_val_transform))

    if args.mode == 'triple' or args.scorer in ['consistency']:
        tf1 = transforms.Compose([ resize_and_crop, transforms.RandomHorizontalFlip() ])
        tf2 = transforms.Compose([ resize_and_crop, RandomAffine(0.0, 0.1) ])
        tf3 = transforms.Compose([ resize_and_crop, GaussianBlur(0.1) ])

        perturbations = [ tf1, tf2, tf3 ]

        perturb_target_train_data = Perturbate(target_train_data,
                                               transform=transforms.Compose(target_val_transform),
                                               perturbations=perturbations,
                                               num_perturbations=len(perturbations))

    print("=> creating source model '{}'".format(args.arch))

    source_model_file = os.path.join(cluster_dir, cluster_prefix + '.pth')

    num_classes = 31

    if 'resnet' in args.arch:
        arch = getattr(models, args.arch)(pretrained=True)
    else:
        raise RuntimeError('A pretrained model must be provided')

    feature_dim = getattr(arch, 'fc').in_features
    setattr(arch, 'fc', nn.Linear(feature_dim, num_classes))
    model = dialnet.DIAL(arch, args.mode)

    if args.mode == 'single':
        criterion = nn.CrossEntropyLoss()
    elif args.mode == 'dual':
        criterion = dialnet.DIALEntropyLoss()
    elif args.mode == 'triple':
        if args.target == 'webcam':
            criterion = dialnet.DIALConsistencyLoss(consistency_loss_weight=0.1)
        else:
            criterion = dialnet.DIALConsistencyLoss(consistency_loss_weight=0.02)

    print("=> clustering target data {} using source model '{}'".format(args.target, args.arch))

    features = []
    classifier = []
    for name, param in model.named_parameters():
        if name.startswith('base_model.fc'):
            classifier.append(param)
        else:
            features.append(param)

    params['optimizer'] = [
        {'params': features},
	{'params': classifier, 'lr': args.lr}
    ]
    params['lr'] = params['lr'] * 0.1

    params['resume'] = source_model_file[:-4] + '_checkpoint.pth.tar'
    cluster = DLCluster(model, criterion, optimizer='SGD', params=params)
    if not os.path.isfile(source_model_file):
        if args.mode == 'dual':
            target_train_data.transform = transforms.Compose(target_train_transform)
            model_weights = cluster.fit(source_train_data, target_train_data, target_val_data)
            target_train_data.transform = None
        elif args.mode == 'triple':
            num_perturbations = perturb_target_train_data.num_perturbations
            perturb_target_train_data.num_perturbations = 1
            model_weights = cluster.fit(source_train_data, perturb_target_train_data, target_val_data)
            perturb_target_train_data.num_perturbations = num_perturbations
        else:
            model_weights = cluster.fit(source_train_data, val_data=target_val_data)
        torch.save(model_weights, source_model_file)

    if args.clustering:
        return

    if args.scorer in ['entropy']:
        target_train_data.transform = transforms.Compose(target_val_transform)
        probs = cluster.predict(target_train_data)
        target_train_data.transform = None
    else:
        probs = cluster.predict(perturb_target_train_data)

    if args.scorer:
        if args.scorer == 'energy':
            scorer = Energy()
        elif args.scorer == 'entropy':
            scorer = Entropy()
        elif args.scorer == 'consistency':
            scorer = Consistency()
        elif args.scorer == 'consensus':
            scorer = Consensus()

    if args.sampler:
        if args.sampler == 'random':
            sampler = Random(ratio=args.sample_ratio)
            selected_indices = sampler.select(len(target_train_data))
        elif args.sampler in ['toprank', 'uniform']:
            if args.sampler == 'toprank':
                sampler = TopRank(scorer, ratio=args.sample_ratio)
            elif args.sampler == 'uniform':
                sampler = Uniform(scorer, ratio=args.sample_ratio)
            selected_indices = sampler.select(probs)
        target_train_data_budget = Sample(target_train_data, indices=selected_indices, transform=transforms.Compose(target_train_transform))
        target_train_data_remain = Sample(target_train_data, indices=[idx for idx in range(len(target_train_data)) if not idx in selected_indices], transform=transforms.Compose(target_train_transform))

    print("=> {} samples were selected from target data '{}'".format(len(target_train_data_budget), args.target))

    run_prefix = cluster_prefix + '_' + \
                 ((args.sampler.lower() + '_') if args.sampler else '') + \
                 ((str(args.sample_ratio) + '_') if args.sampler else '') + \
                 ((args.scorer.lower() + '_') if args.scorer else '') + \
                 ('finetune_' if args.pretrained else 'scratch_') + \
                 ('useall' if args.useall else 'budget')
    run_dir = os.path.join('runs', run_prefix)

    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
        if tblog.is_enabled:
            tblog.logdir = run_dir

    if tblog.is_enabled and args.scorer and args.sampler in ['random', 'toprank', 'uniform']:
        scores = scorer.compute(probs)
        with tblog.SummaryWriter(tblog.logdir) as writer:
            writer.add_histogram('Scores/train', scores)
            writer.add_histogram('Scores/sample', scores[selected_indices])

    print("=> creating target model '{}'".format(args.arch))

    if 'resnet' in args.arch:
        arch = getattr(models, args.arch)(pretrained=True)
    else:
        raise RuntimeError('A pretrained model must be provided')

    feature_dim = getattr(arch, 'fc').in_features
    setattr(arch, 'fc', nn.Linear(feature_dim, num_classes))
    model = dialnet.DIAL(arch, 'single')

    if args.pretrained and os.path.isfile(source_model_file):
        state_dict = torch.load(source_model_file)
        if 'base_model.fc.bias' in state_dict.keys() and \
            state_dict['base_model.fc.bias'].size(0) != num_classes:
            state_dict.pop('base_model.fc.bias')
        if 'base_model.fc.weight' in state_dict.keys() and \
            state_dict['base_model.fc.weight'].size(0) != num_classes:
            state_dict.pop('base_model.fc.weight')
        model.load_state_dict(state_dict, strict=False)

    target_model_file = os.path.join(run_dir, run_prefix + '.pth')

    if args.useall:
        criterion = dialnet.DIALEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    print("=> training on budget data {} using target model '{}'".format(args.target, args.arch))

    if args.pretrained:
        params['lr'] = args.lr * 0.1
    params['resume'] = target_model_file[:-4] + '_checkpoint.pth.tar'
    classifier = DLClassifier(model, criterion, optimizer='SGD', params=params)

    if args.evaluate:
        with torch.no_grad():
            classifier.predict(target_val_data, accuracy=True)
        return

    if args.useall:
        model_weights = classifier.fit(target_train_data_budget, target_train_data_remain, target_val_data)
    else:
        model_weights = classifier.fit(target_train_data_budget, val_data=target_val_data)
    torch.save(model_weights, target_model_file)

if __name__ == '__main__':
    main()
