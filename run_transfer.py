import os
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import scipy.io

from torchvision import datasets, transforms

from models import OneLayerMLP 
from data import get_data
from vgg import VGG16_conv
from res


parser = ArgumentParser()
parser.add_argument('--transfer_dataset', default='caltech', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--data', default='../data')
parser.add_argument('--alg', required=True, choices=['vgg', 'res', ])
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--model_path', required=True, type=str, help='Path to model path')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()

# NOTE: THESE ARGUMENTS DON'T MATTER
args.ssl = False
args.std = 0
args.std_factor = 0.9
args.epoch = 5
args.kernel_size = 3

def test(model, vgg, test_data):
    model.eval()
    total, correct = 0, 0

    for images, labels in test_data:
        if args.cuda:
            images = images.cuda()

        with torch.no_grad():
            preds = vgg(images, return_intermediate=False)

            preds = torch.argmax(preds, dim=1).cpu().numpy()

            correct += accuracy_score(labels, preds, normalize=False)
            total += images.size(0)

    model.train()
    return correct / total * 100


args.input_dim = 512
args.dataset = args.transfer_dataset
train_data, test_data, args = get_data(args)

model = OneLayerMLP(input_dim=args.input_dim, num_classes=args.num_classes)
optim_model = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

vgg = VGG16_conv(n_classes=args.num_classes, args=args)

weights = torch.load(args.model_path)
vgg.load_state_dict(weights)

ce_loss = F.cross_entropy

best_acc = 0

if args.cuda:
    vgg = vgg.cuda()
    model = model.cuda()

for epoch_count in range(args.num_epochs):
    for count, (img, label) in enumerate(test_data):
        if args.cuda:
            img = img.cuda()
            label = label.cuda()

        with torch.no_grad():
            feature = vgg(img, return_intermediate=True)

        preds = model(feature)
        loss = ce_loss(preds, label.view(-1))

        optim_model.zero_grad()
        loss.backward()
        optim_model.step()
        if count % 100 == 0:
            print('loss: ', loss.item())

    acc = test(model, vgg, test_data)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch_count

    print('curr epoch: {} curr acc: {}'.format(epoch_count, acc))
    print('best epoch: {} best acc: {}'.format(best_epoch, best_acc))
