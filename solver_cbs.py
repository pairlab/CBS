import torch
import copy
import os
import torch.optim as optim
import torch.utils.data as data
import math

from arguments import get_args

from models import *
from data import get_data
from solver_base import BaseSolver

class CBSSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

        self.decay_epoch = 50 if self.args.alg == 'vgg' else 30
        self.stop_decay_epoch = self.decay_epoch * 3 + 1

    def solve(self):
        best_epoch, best_acc = 0, 0
        num_iter = 0
        for epoch_count in range(self.args.num_epochs):
            self.model.get_new_kernels(epoch_count)
        
            if self.cuda:
                self.model = self.model.cuda()

            if epoch_count is not 0 and epoch_count % self.decay_epoch == 0 \
                    and epoch_count < self.stop_decay_epoch:
                for param in self.optim.param_groups:
                    param['lr'] = param['lr'] / 10 

            for images, labels in self.train_data:
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                preds = self.model(images)
                loss = self.ce_loss(preds, labels)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step() 

                num_iter += 1

                if num_iter % 200 == 0:
                    print('iter num: {} \t loss: {:.2f}'.format(
                            num_iter, loss.item()))

            if epoch_count % 1 == 0:
                accuracy = self.test()
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_epoch = epoch_count
                    self.best_model = copy.deepcopy(self.model)

                print('epoch count: {} \t accuracy: {:.2f}'.format(
                        epoch_count, accuracy))
                print('best acc: {} \t best acc: {:.2f}'.format(
                        best_epoch, best_acc))
