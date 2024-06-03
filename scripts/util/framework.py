import os
import sklearn.metrics
import numpy as np
import sys
import time
from util.sentence_encoder import CNNSentenceEncoder
from util.data_loader import get_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing. !!! not yet included in current module
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
    def train_maml(self, model, maml, prefix, batch_size, trainN, N, K, Q,
                   pytorch_optim=optim.SGD, load_ckpt=None, save_ckpt=None,
                   na_rate=0, val_step=2000,
                   train_iter=30000, val_iter=1000,
                   learning_rate=1e-1, grad_iter=1):
        if torch.cuda.is_available():
            model.cuda()

        optimizer = pytorch_optim(maml.parameters(), lr=learning_rate)

        if load_ckpt:
            state_dict = torch.load(load_ckpt)['state_dict']
            model.load_state_dict(state_dict)

        for iteration in range(train_iter):
            support_set, support_labels, query_set, query_labels = next(iter(self.train_data_loader))

            if torch.cuda.is_available():
                support_set, support_labels = support_set.cuda(), support_labels.cuda()
                query_set, query_labels = query_set.cuda(), query_labels.cuda()

            maml.train()
            optimizer.zero_grad()
            loss, _ = maml(support_set, support_labels, query_set, query_labels)
            loss.backward()
            optimizer.step()

            if (iteration + 1) % val_step == 0:
                maml.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                for val_support_set, val_support_labels, val_query_set, val_query_labels in self.val_data_loader:
                    if torch.cuda.is_available():
                        val_support_set, val_support_labels = val_support_set.cuda(), val_support_labels.cuda()
                        val_query_set, val_query_labels = val_query_set.cuda(), val_query_labels.cuda()

                    with torch.no_grad():
                        loss, preds = maml(val_support_set, val_support_labels, val_query_set, val_query_labels)
                        val_loss += loss.item()
                        correct += (preds.argmax(1) == val_query_labels).sum().item()
                        total += val_query_labels.size(0)

                print("Validation Loss: {:.4f}, Accuracy: {:.4f}".format(val_loss / len(self.val_data_loader), correct / total))
                torch.save({'state_dict': model.state_dict()}, save_ckpt)
