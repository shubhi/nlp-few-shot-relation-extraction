import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
#from torch.utils.data import DataLoader, Dataset

class RelationDataset(data.Dataset):
    def __init__(self, data, word2id, max_length):
        self.data = data
        self.word2id = word2id
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.data[index]
        tokens = item['tokens']
        token_ids = [self.word2id.get(token, self.word2id['[UNK]']) for token in tokens]
        token_ids = token_ids[:self.max_length]
        token_ids += [self.word2id['[PAD]']] * (self.max_length - len(token_ids))
        return torch.tensor(token_ids), torch.tensor(item['relation'])

    def __len__(self):
        return len(self.data)

def get_loader(file_name, encoder_name, N, K, Q, batch_size, max_length, glove_mat, glove_word2id):
    with open(file_name, 'r') as f:
        data = json.load(f)

    def collate_fn(data):
        tokens, relations = zip(*data)
        tokens = torch.stack(tokens)
        relations = torch.stack(relations)
        return tokens, relations

    dataset = RelationDataset(data, glove_word2id, max_length)
    loader = data.DataLoader(dataset, batch_size=N*(K+Q), shuffle=True, collate_fn=collate_fn)

    def get_task_batch():
        batch = next(iter(loader))
        tokens, relations = batch
        support_set = (tokens[:N*K], relations[:N*K])
        query_set = (tokens[N*K:], relations[N*K:])
        return support_set, query_set

    loader.get_task_batch = get_task_batch

    return loader

class FewRelDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        print("Loading data from {} ...".format(path))
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        # Clip indices to max allowed values
        word = np.clip(word, 0, 400000)
        pos1 = np.clip(pos1, 0, 255)
        pos2 = np.clip(pos2, 0, 255)
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q

        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label

def get_loader(name, encoder, N, K, Q, batch_size,
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    print("Dataset size:", len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    print("Data loader size:", len(data_loader))
    return iter(data_loader)

def data_loader_verify(data_loader, num_batches=2):
    for i in range(num_batches):
        try:
            support_set, query_set, query_label = next(data_loader)
            print(f"Batch {i+1}:")
            print("Support set word shape:", support_set['word'].shape)
            print("Support set pos1 shape:", support_set['pos1'].shape)
            print("Support set pos2 shape:", support_set['pos2'].shape)
            print("Support set mask shape:", support_set['mask'].shape)
            print("Query set word shape:", query_set['word'].shape)
            print("Query set pos1 shape:", query_set['pos1'].shape)
            print("Query set pos2 shape:", query_set['pos2'].shape)
            print("Query set mask shape:", query_set['mask'].shape)
            print("Query label shape:", query_label.shape)
            print("Query labels:", query_label)
            print("-" * 50)
        except StopIteration:
            print("No more data available from the data loader.")
            break