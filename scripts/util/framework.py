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
        # self.test_data_loader = test_data_loader
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model, model_name,
              B, N_for_train, N_for_eval, K, Q, na_rate=0,
              learning_rate=1e-1, lr_step_size=20000, weight_decay=1e-5,
              train_iter=30000, val_iter=1000, val_step=2000, test_iter=3000,
              load_ckpt=None, save_ckpt=None,
              pytorch_optim=optim.SGD,
              grad_iter=1):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
    
        # Init
        
        optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            support, query, label = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()

            logits, pred = model(support, query, 
                    N_for_train, K, Q * N_for_train + na_rate * Q)
            loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, 
                        na_rate=na_rate)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset !!!! module not setup yet")
            # if ckpt != 'none':
            #     state_dict = self.__load_model__(ckpt)['state_dict']
            #     own_state = model.state_dict()
            #     for name, param in state_dict.items():
            #         if name not in own_state:
            #             continue
            #         own_state[name].copy_(param)
            # eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, label = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                logits, pred = model(support, query, N, K, Q * N + Q * na_rate)

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample