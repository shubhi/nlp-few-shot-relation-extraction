import os
import sklearn.metrics
import numpy as np
import sys
import time
import torch
import matplotlib.pyplot as plt
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from util.sentence_encoder import CNNSentenceEncoder
from util.data_loader import get_loader

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
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

    def train(self, model, prefix, batch_size, trainN, N, K, Q,
              pytorch_optim, load_ckpt, save_ckpt, na_rate, val_step,
              train_iter, val_iter, learning_rate, grad_iter, output_dir):
        
        optimizer = pytorch_optim(model.parameters(), lr=learning_rate)
        if load_ckpt:
            state_dict = torch.load(load_ckpt)['state_dict']
            model.load_state_dict(state_dict)
        
        best_acc = 0
        model.train()
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for it in range(train_iter):
            support, query, label = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()

            logits, pred = model(support, query, N, K, Q * N + na_rate * Q)
            loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if (it + 1) % grad_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

            iter_loss += loss.item()
            iter_right += right.item()
            iter_sample += 1
            train_losses.append(iter_loss / iter_sample)
            train_accuracies.append(100 * iter_right / iter_sample)
            print('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample), end='\r')

            if (it + 1) % val_step == 0:
                val_loss, acc = self.eval(model, batch_size, N, K, Q, val_iter, na_rate=na_rate)
                val_losses.append(val_loss)
                val_accuracies.append(acc)
                print("\n[EVAL] step: {} | accuracy: {:.2f}%".format(it + 1, 100 * acc))
                if acc > best_acc:
                    print("Best checkpoint")
                    best_acc = acc
                    torch.save({
                        'state_dict': model.state_dict()
                    }, save_ckpt)
        
        self.log_and_visualize(train_losses, train_accuracies, val_losses, val_accuracies, output_dir)

    def eval(self, model, batch_size, N, K, Q, eval_iter, na_rate=0):
        model.eval()

        iter_right = 0.0
        iter_sample = 0.0
        iter_loss = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, label = next(self.val_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()

                logits, pred = model(support, query, N, K, Q * N + na_rate * Q)
                loss = model.loss(logits, label)
                right = model.accuracy(pred, label)

                iter_loss += loss.item()
                iter_right += right.item()
                iter_sample += 1

        model.train()
        return iter_loss / iter_sample, iter_right / iter_sample

    def log_and_visualize(self, train_losses, train_accuracies, val_losses, val_accuracies, output_dir):
        """
        Logs and visualizes the training loss, training accuracy, and validation accuracy.

        Args:
        - train_losses (list): List of training losses.
        - train_accuracies (list): List of training accuracies.
        - val_losses (list): List of validation losses.
        - val_accuracies (list): List of validation accuracies.
        - output_dir (str): Directory to save the plots and logs.
        """

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the training logs
        np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(output_dir, 'train_accuracies.npy'), np.array(train_accuracies))
        np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))
        np.save(os.path.join(output_dir, 'val_accuracies.npy'), np.array(val_accuracies))

        # Plotting the loss
        plt.figure()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss.png'))

        # Plotting the accuracy
        plt.figure()
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'accuracy.png'))

        print(f"Plots saved in {output_dir}")