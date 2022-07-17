from __future__ import print_function
import os
import random
import os.path
import numpy as np

import torch
import torchvision
if torchvision.__version__ < '0.3.0':
    from torch.utils.data import Dataset as VisionDataset
else:
    from torchvision.datasets.vision import VisionDataset

from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets.folder import default_loader

if torchvision.__version__ < '0.4.0':
    def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
        if not isinstance(value, torch._six.string_classes):
            if arg is None:
                msg = "Expected type str, but got type {type}."
            else:
                msg = "Expected type str for argument {arg}, but got type {type}."
            msg = msg.format(type=type(value), arg=arg)
            raise ValueError(msg)

        if valid_values is None:
            return value

        if value not in valid_values:
            if custom_msg is not None:
                msg = custom_msg
            else:
                msg = ("Unknown value '{value}' for argument {arg}. "
                       "Valid values are {{{valid_values}}}.")
                msg = msg.format(value=value, arg=arg,
                                 valid_values=iterable_to_str(valid_values))
            raise ValueError(msg)

        return value
else:
    from torchvision.datasets.utils import verify_str_arg

from transforms import Map


class CIFAR9(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR9, self).__init__(root, train=train, transform=transform,
                                     target_transform=Map({0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 7:6, 8:7, 9:8}),
                                     download=download)
        self.indices = np.where(np.array(self.targets) != 6)[0]
        self._target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super(CIFAR9, self).__getitem__(self.indices[index])

        if self._target_transform is not None:
            target = self._target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)


class STL9(STL10):
    splits = ('train', 'test')

    def __init__(self, root, split='train', folds=None, transform=None,
                 target_transform=None, download=False):
        self.split = verify_str_arg(split, "split", self.splits)
        super(STL9, self).__init__(root, split=split, folds=folds, transform=transform,
                                   target_transform=Map({0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 8:7, 9:8}),
                                   download=download)
        self.indices = np.where(self.labels != 7)[0]
        self._target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super(STL9, self).__getitem__(self.indices[index])

        if self._target_transform is not None:
            target = self._target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)


def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath, imlabel = line.strip().split()
			imlist.append( (impath, int(imlabel)) )

	return imlist


class ImageList(VisionDataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):

        if torchvision.__version__ < '0.3.0':
            self.root = os.path.expanduser(dataset.root)
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(ImageList, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        if not isinstance(flist, list):
            if flist_reader is not None:
                self.imlist = flist_reader(flist)
            else:
                raise RuntimeError('A list of images and their labels must be provided')
        else:
            self.imlist = flist

        if loader is not None:
            self.loader = loader
        else:
            raise RuntimeError('A function to load images must be provided')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


class Repeat(VisionDataset):
    def __init__(self, dataset, transform=None, target_transform=None,
                 repeats=1):

        if torchvision.__version__ < '0.3.0':
            self.root = os.path.expanduser(dataset.root)
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(Repeat, self).__init__(dataset.root, transform=transform,
                                         target_transform=target_transform)

        if not isinstance(dataset, VisionDataset):
             raise RuntimeError('A dataset must be provided')

        self.dataset = dataset
        self.repeats = repeats
        self.indices = np.repeat(range(len(dataset)), repeats, axis=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[self.indices[index]]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)


class Sample(VisionDataset):
    def __init__(self, dataset, transform=None, target_transform=None,
                 indices=None):

        if torchvision.__version__ < '0.3.0':
            self.root = os.path.expanduser(dataset.root)
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(Sample, self).__init__(dataset.root, transform=transform,
                                          target_transform=target_transform)

        if not isinstance(dataset, VisionDataset):
             raise RuntimeError('A dataset must be provided')

        self.dataset = dataset
        if indices is not None:
            self.indices = indices
        else:
            self.indices = range(len(dataset))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[self.indices[index]]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)


class Subset(Sample):
    def __init__(self, dataset, transform=None, target_transform=None,
                 proportions=None):

        if not isinstance(dataset, VisionDataset):
             raise RuntimeError('A dataset must be provided')

        targets = []
        for _, target in dataset:
            targets.append(target)
        unique_targets, counts = np.unique(targets, return_counts=True)

        if proportions is not None:
            self.proportions = proportions

            indices = []
            for target, count in zip(unique_targets, counts):
                size = int(self.proportions[target] * count)
                target_indices = np.where(targets == target)[0]
                chosen_indices = np.random.choice(target_indices, size=size, replace=False)
                indices.extend(chosen_indices)
        else:
            self.proportions = counts / len(dataset)
            indices = None

        super(Subset, self).__init__(dataset, transform=transform,
                                     target_transform=target_transform,
                                     indices=indices)


class Perturbate(VisionDataset):
    def __init__(self, dataset, transform=None, target_transform=None,
                 perturbations=None, num_perturbations=0):

        if torchvision.__version__ < '0.3.0':
            self.root = os.path.expanduser(dataset.root)
            self.transform = transform
            self.target_transform = target_transform
        else:
            super(Perturbate, self).__init__(dataset.root, transform=transform,
                                             target_transform=target_transform)

        if not isinstance(dataset, VisionDataset):
             raise RuntimeError('A dataset must be provided')

        self.dataset = dataset
        self.perturbations = perturbations
        self.num_perturbations = num_perturbations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        imgs = [img]
        if self.perturbations is not None:
            selected_perturbations = random.sample(self.perturbations,
                                                   self.num_perturbations)
            for perturbation in selected_perturbations:
                imgs.append(perturbation(img))

        if self.transform is not None:
            for i, img in enumerate(imgs):
                imgs[i] = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.stack(imgs), target

    def __len__(self):
        return len(self.dataset)

