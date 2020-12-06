import abc
import os
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from resnet import *
from vgg import VGG16_conv
from data import get_data
import wide_resnet


class BaseSolver(metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.train_data, self.test_data, self.args = get_data(args)
        self.cuda = torch.cuda.is_available()

        if self.args.alg == 'normal':
            self.model = CNNNormal(
                    nc=self.args.in_dim,
                    num_classes=self.args.num_classes,
            )
        elif self.args.alg == 'vgg':
            self.model = VGG16_conv(
                    self.args.num_classes,
                    args=args,
            )
        elif self.args.alg == 'res':
            self.model = ResNet18(self.args)
        elif self.args.alg == 'wrn':
            self.model = wide_resnet.Wide_ResNet(52, 2, 0.3, self.args.num_classes, args)

        self.optim = optim.SGD(
                self.model.parameters(), 
                lr=args.lr,
                weight_decay=5e-4, 
                momentum=0.9,
        )

        if self.cuda:
            self.model.cuda()

        self.ce_loss = F.cross_entropy


    def test(self):
        self.model.eval()
        total, correct = 0, 0
        for images, labels in self.test_data:
            if self.cuda:
                images = images.cuda()

            with torch.no_grad():
                preds = self.model(images)
                preds = torch.argmax(preds, dim=1).cpu().numpy()

                correct += accuracy_score(labels, preds, normalize=False)
                total += images.size(0)

        self.model.train()
        return correct / total * 100


    def save_model(self):
        if not os.path.exists('./weights'):
            os.mkdir('weights/')
    
        filename = os.path.join('weights', self.args.log_name + '_model.tar')
        torch.save(self.best_model.state_dict(), filename)


    @abc.abstractmethod
    def solve(self):
        pass
