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

from model import Model
from transforms import Map, GaussianNoise, GaussianBlur, RandomRotation, RandomAffine
from datasets import CIFAR9, STL9, Perturbate, Repeat, Sample, Subset
from sampler import Random, TopRank, Uniform
from scorer import Energy, Entropy, Consistency, Consensus
from cluster import DLCluster
from classifier import DLClassifier
from utils import Range, PathType

import tblog

if tblog.is_enabled:
    from tqdm import tqdm

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(models.__dict__[name]))
model_names = model_names + ['dialnet']

training_modes = ['single', 'dual', 'triple']

dataset_names = ['CIFAR9', 'STL9', 'MNIST', 'SVHN', 'USPS']

sampler_names = ['random', 'toprank', 'uniform']

scorer_names = ['energy', 'entropy', 'consistency', 'consensus']


def main():
    parser = argparse.ArgumentParser(description='PyTorch CoDIAL + UNFOLD Digits')
    parser.add_argument('--root', '-r', type=PathType(exists=True, type='dir'),
                        metavar='PATH', default='data/Digits',
                        help='path to the Digits dataset')
    parser.add_argument('--source', '-s', metavar='DATA', default='',
                        choices=dataset_names + ['ImageNet'],
                        help='image datasets: ' + ' | '.join(dataset_names +
                        ['ImageNet']) + ' (default: CIFAR9)')
    parser.add_argument('--target', '-t', metavar='DATA', default='STL9',
                        choices=dataset_names,
                        help='image datasets: ' + ' | '.join(dataset_names) +
                        ' (default: STL9)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dialnet',
                        choices=model_names,
                        help='model architectures: ' + ' | '.join(model_names) +
                        ' (default: dialnet)')
    parser.add_argument('--mode', '-m', metavar='MODE', default='dual',
                        choices=training_modes,
                        help='training modes: ' + ' | '.join(training_modes) +
                        ' (default: dual)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run (default: 120)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--lr-steps', default=[50, 90], type=int, nargs="+",
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

    resize = transforms.Resize(32)
    source_train_transform.append(resize)
    target_train_transform.append(resize)
    target_val_transform.append(resize)

    # optionally train on source
    if args.source in ['CIFAR9', 'STL9', 'MNIST', 'SVHN', 'USPS']:
        print("=> preparing source data '{}'".format(args.source))

        source_train_transform.append(transforms.ToTensor())

        if args.source == 'CIFAR9':
            source_normalize = transforms.Normalize(mean=[0.424, 0.415, 0.384],
                                                    std=[0.283, 0.278, 0.284])
            source_train_transform.append(source_normalize)
            source_train_data = CIFAR9(root=args.root, train=True, transform=transforms.Compose(source_train_transform),
                                       target_transform=Map({0:0, 1:2, 2:1, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}) if args.target == 'STL9' else None, download=True)
        elif args.source == 'STL9':
            source_normalize = transforms.Normalize(mean=[0.447, 0.440, 0.407],
                                                    std=[0.260, 0.257, 0.271])
            source_train_transform.append(source_normalize)
            source_train_data = Repeat(STL9(root=args.root, split='train', transform=transforms.Compose(source_train_transform),
                                       target_transform=Map({0:0, 1:2, 2:1, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}) if args.target == 'CIFAR9' else None, download=True), repeats=10)
        elif args.source == 'MNIST':
            # to make MNIST RGB instead of grayscale
            source_gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            source_normalize = transforms.Normalize(mean=[0.5] if args.target == 'USPS' else [0.5, 0.5, 0.5],
                                                    std=[0.5] if args.target == 'USPS' else [0.5, 0.5, 0.5])
            source_train_transform.extend([source_normalize] if args.target == 'USPS' else [source_gray2rgb, source_normalize])
            source_train_data = datasets.__dict__[args.source](root=args.root, train=True, transform=transforms.Compose(source_train_transform), download=True)
        elif args.source == 'SVHN':
            source_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5])
            source_train_transform.append(source_normalize)
            source_train_data = datasets.__dict__[args.source](root=args.root + '/SVHN', split='train', transform=transforms.Compose(source_train_transform), download=True)
        elif args.source == 'USPS':
            source_normalize = transforms.Normalize(mean=[0.5],
                                                    std=[0.5])
            source_train_transform.append(source_normalize)
            source_train_data = datasets.__dict__[args.source](root=args.root, train=True, transform=transforms.Compose(source_train_transform), download=True)

    print("=> preparing target data '{}'".format(args.target))

    target_train_transform.append(transforms.ToTensor())

    target_val_transform.append(transforms.ToTensor())

    if args.target == 'CIFAR9':
        target_normalize = transforms.Normalize(mean=[0.424, 0.415, 0.384],
                                                std=[0.283, 0.278, 0.284])
        target_train_transform.append(target_normalize)
        target_val_transform.append(target_normalize)
        target_train_data = Subset(CIFAR9(root=args.root, train=True, download=True), proportions=args.subset)
        target_val_data = CIFAR9(root=args.root, train=False, transform=transforms.Compose(target_val_transform), download=True)
    elif args.target == 'STL9':
        target_normalize = transforms.Normalize(mean=[0.447, 0.440, 0.407],
                                                std=[0.260, 0.257, 0.271])
        target_train_transform.append(target_normalize)
        target_val_transform.append(target_normalize)
        target_train_data = Subset(STL9(root=args.root, split='train', download=True), proportions=args.subset)
        target_val_data = STL9(root=args.root, split='test', transform=transforms.Compose(target_val_transform), download=True)
    elif args.target == 'MNIST':
        # to make MNIST RGB instead of grayscale
        target_gray2rgb = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        target_normalize = transforms.Normalize(mean=[0.5] if args.source == 'USPS' else [0.5, 0.5, 0.5],
                                                std=[0.5] if args.source == 'USPS' else [0.5, 0.5, 0.5])
        target_train_transform.extend([target_normalize] if args.source == 'USPS' else [target_gray2rgb, target_normalize])
        target_val_transform.extend([target_normalize] if args.source == 'USPS' else [target_gray2rgb, target_normalize])
        target_train_data = Subset(datasets.__dict__[args.target](root=args.root, train=True, download=True), proportions=args.subset)
        target_val_data = datasets.__dict__[args.target](root=args.root, train=False, transform=transforms.Compose(target_val_transform), download=True)
    elif args.target == 'SVHN':
        target_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
        target_train_transform.append(target_normalize)
        target_val_transform.append(target_normalize)
        target_train_data = Subset(datasets.__dict__[args.target](root=args.root + '/SVHN', split='train', download=True), proportions=args.subset)
        target_val_data = datasets.__dict__[args.target](root=args.root + '/SVHN', split='test', transform=transforms.Compose(target_val_transform), download=True)
    elif args.target == 'USPS':
        target_normalize = transforms.Normalize(mean=[0.5],
                                                std=[0.5])
        target_train_transform.append(target_normalize)
        target_val_transform.append(target_normalize)
        target_train_data = Subset(datasets.__dict__[args.target](root=args.root, train=True, download=True), proportions=args.subset)
        target_val_data = datasets.__dict__[args.target](root=args.root, train=False, transform=transforms.Compose(target_val_transform), download=True)

    if args.mode == 'triple' or args.scorer in ['consistency', 'consensus']:
        tf1 = transforms.Compose([ resize, transforms.RandomHorizontalFlip() ])
        tf2 = transforms.Compose([ resize, transforms.RandomCrop(32, 4) ])
        tf3 = transforms.Compose([ resize, RandomAffine(0.0, 0.1) ])
        tf4 = transforms.Compose([ resize, GaussianBlur(0.1) ])

        if args.target in ['CIFAR9', 'STL9']:
            perturbations = [ tf1, tf2, tf3, tf4 ]
        elif args.target in ['MNIST', 'SVHN', 'USPS']:
            perturbations = [      tf2, tf3, tf4 ]

        perturb_target_train_data = Perturbate(target_train_data,
                                               transform=transforms.Compose(target_val_transform),
                                               perturbations=perturbations,
                                               num_perturbations=len(perturbations))

    print("=> creating source model '{}'".format(args.arch))

    source_model_file = os.path.join(cluster_dir, cluster_prefix + '.pth')

    if args.source in ['CIFAR9', 'STL9', 'MNIST', 'SVHN', 'USPS']:
        if args.source in ['CIFAR9', 'STL9'] and args.target in ['CIFAR9', 'STL9']:
            num_classes = 9
            arch = 'cifar9-stl9'
        elif args.source in ['MNIST', 'SVHN'] and args.target in ['MNIST', 'SVHN']:
            num_classes = 10
            arch = 'mnist-svhn'
        elif args.source in ['MNIST', 'USPS'] and args.target in ['MNIST', 'USPS']:
            num_classes = 10
            arch = 'mnist-usps'
        else:
            raise NotImplementedError

        model_params = dict()
        model_params['training_mode'] = args.mode
        model_params['arch'] = arch

        model = Model(num_classes, base_model=args.arch, model_file=source_model_file, **model_params)
    elif 'resnet' in args.arch:
        model = getattr(models, args.arch)(pretrained=True)
    else:
        raise RuntimeError('A pretrained model must be provided')

    if 'resnet' in args.arch:
        criterion = nn.CrossEntropyLoss()
    elif 'dialnet' in args.arch:
        if args.mode == 'single':
            criterion = nn.CrossEntropyLoss()
        elif args.mode == 'dual':
            criterion = dialnet.DIALEntropyLoss()
        elif args.mode == 'triple':
            if args.source == 'CIFAR9' and args.target in 'STL9':
                consistency_loss_weight = 0.1
            elif args.source in ['MNIST', 'USPS'] and args.target in ['MNIST', 'USPS']:
                consistency_loss_weight = 0.1
            else:
                consistency_loss_weight = 0.02
            criterion = dialnet.DIALConsistencyLoss(consistency_loss_weight=consistency_loss_weight)

    print("=> clustering target data {} using source model '{}'".format(args.target, args.arch))

    params['resume'] = source_model_file[:-4] + '_checkpoint.pth.tar'
    cluster = DLCluster(model, criterion, optimizer='Adam', params=params)
    if args.source in ['CIFAR9', 'STL9', 'MNIST', 'SVHN', 'USPS'] and not os.path.isfile(source_model_file):
        if 'dialnet' in args.arch and args.mode in ['dual', 'triple']:
            if args.mode == 'dual':
                target_train_data.transform = transforms.Compose(target_train_transform)
                model_weights = cluster.fit(source_train_data, Repeat(target_train_data, repeats=10) if args.target == 'STL9' else target_train_data, target_val_data)
                target_train_data.transform = None
            elif args.mode == 'triple':
                num_perturbations = perturb_target_train_data.num_perturbations
                perturb_target_train_data.num_perturbations = 1
                model_weights = cluster.fit(source_train_data, Repeat(perturb_target_train_data, repeats=10) if args.target == 'STL9' else perturb_target_train_data, target_val_data)
                perturb_target_train_data.num_perturbations = num_perturbations
        else:
            model_weights = cluster.fit(source_train_data, val_data=target_val_data)
        torch.save(model_weights, source_model_file)

    if args.clustering:
        return

    if args.scorer in ['energy', 'entropy']:
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

    if args.target in ['CIFAR9', 'STL9']:
        num_classes = 9
    elif args.target in ['MNIST', 'SVHN', 'USPS']:
        num_classes = 10

    print("=> creating target model '{}'".format(args.arch))

    model_params = dict()
    if 'resnet' in args.arch:
        model_params['pretrained'] = args.pretrained
    elif 'dialnet' in args.arch:
        model_params['training_mode'] = 'single'
        model_params['arch'] = arch
    model = Model(num_classes, base_model=args.arch, model_file=source_model_file if args.pretrained else '', **model_params)
    target_model_file = os.path.join(run_dir, run_prefix + '.pth')

    if args.useall:
        criterion = dialnet.DIALEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    print("=> training on budget data {} using target model '{}'".format(args.target, args.arch))

    if args.pretrained:
        params['lr'] = params['lr'] * 0.1
    params['resume'] = target_model_file[:-4] + '_checkpoint.pth.tar'
    classifier = DLClassifier(model, criterion, optimizer='Adam', params=params)

    if args.evaluate:
        with torch.no_grad():
            classifier.predict(target_val_data, accuracy=True)
        return

    if args.useall:
        model_weights = classifier.fit(target_train_data_budget, target_train_data_remain, target_val_data)
    else:
        model_weights = classifier.fit(target_train_data_budget, val_data=target_val_data)
    torch.save(model_weights, target_model_file)

    if tblog.is_enabled:
        inputs, targets, outputs = extract_features(target_val_data, model, 'avgpool')

        with tblog.SummaryWriter(tblog.logdir) as writer:
            writer.add_embedding(
                outputs,
                metadata=targets,
                label_img=inputs)

def extract_features(data, model, layer_name):
    inputs = []
    targets = []
    outputs = []

    layer = getattr(getattr(model, 'base_model'), layer_name)
    if not isinstance(layer, nn.Sequential):
        def hook_fn(model, input, output):
            outputs.extend(output.squeeze().cpu())

        layer.register_forward_hook(hook_fn)

        loader = DataLoader(data,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

        model = torch.nn.DataParallel(model).cuda()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            with tqdm(total=len(loader), ascii=True, desc='extracting features') as pbar:
                for i, (input, target) in enumerate(loader):
                    input_var = input # torch.autograd.Variable(input, volatile=True).cuda()

                    # compute output
                    output = model(input_var)

                    targets.extend(target)
                    inputs.extend(input)

                    pbar.update(1)

    return torch.stack(inputs), torch.stack(targets), torch.stack(outputs)

if __name__ == '__main__':
    main()
