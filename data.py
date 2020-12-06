import os
import scipy.io
import numpy as np
import random

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

def get_data(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))]
        )
    
        train_data = datasets.MNIST(
                root=args.data,
                download=True,
                train=True,
                transform=transform
        )
        test_data = datasets.MNIST(
                root=args.data,
                download=True,
                train=False,
                transform=transform
        )
        args.num_classes = 10
        args.in_dim = 28*28 
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
    
        train_data = datasets.CIFAR10(
                root=args.data,
                download=True,
                train=True,
                transform=transform
        )
        test_data = datasets.CIFAR10(
                root=args.data,
                download=True,
                train=False,
                transform=transform
        )
        args.num_classes = 10
        args.in_dim = 3
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
    
        train_data = datasets.CIFAR100(
                root=args.data,
                download=True,
                train=True,
                transform=transform
        )
        test_data = datasets.CIFAR100(
                root=args.data,
                download=True,
                train=False,
                transform=transform
        )
        args.num_classes = 100
        args.in_dim = 3
    elif args.dataset == 'imagenet':
        transform = transforms.Compose([
                transforms.Scale(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
        train_data = datasets.ImageFolder(
            os.path.join(args.data, 'tiny-imagenet-200', 'train'),
            transform=transform,
        )
        test_data = datasets.ImageFolder(
            os.path.join(args.data, 'tiny-imagenet-200', 'val'),
            transform=transform,
        )
        args.num_classes = 200
        args.in_dim = 3
    elif args.dataset == 'svhn':      
        transform = transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
    
        train_data = datasets.SVHN(
                root=args.data,
                download=True,
                split='train',
                transform=transform
        )
        test_data = datasets.SVHN(
                root=args.data,
                download=True,
                split='test',
                transform=transform
        )
        args.num_classes = 10
        args.in_dim = 3
    elif args.dataset == 'caltech':
        args.num_classes = 101
        transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.Scale(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
        test_transform = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.Scale(64),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ]
        )
        train_data = datasets.Caltech101(
                root=args.data,
                download=False,
                transform=transform,
        )
        test_data = datasets.Caltech101(
                root=args.data,
                download=False,
                transform=test_transform,
        )
    else:
        raise NotImplementedError


    if args.ssl:
        all_indices = [i for i in range(len(train_data))]
        indices = random.sample(all_indices, int(args.percentage*len(train_data)/100))
    
        sampler = data.sampler.SubsetRandomSampler(indices)        
        train_loader = data.DataLoader(
                train_data, 
                batch_size=args.batch_size,
                pin_memory=True,
                num_workers=int(4),
                shuffle=False,
                drop_last=True,
                sampler=sampler
        )

    else:
        train_loader = data.DataLoader(
                train_data, 
                batch_size=args.batch_size,
                pin_memory=True,
                num_workers=int(4),
                shuffle=True,
                drop_last=True,
        )


    test_loader = data.DataLoader(
            test_data, 
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=int(4),
            shuffle=True,
            drop_last=False,
    )

    return train_loader, test_loader, args

