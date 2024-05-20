import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vectors


class FewRelDataset(Dataset):
    def __init__(self, data_path, glove_vectors, max_seq_len=128):
        self.data = self.load_data(data_path)
        self.glove_vectors = glove_vectors
        self.max_seq_len = max_seq_len
        self.relations = list(self.data.keys())
        self.label_encoder = LabelEncoder().fit(self.relations)
        self.samples = self.create_samples()

    def load_data(self, data_path):
        with open(data_path, 'r') as file:
            data = json.load(file)
        return data

    def tokenize_and_vectorize(self, tokens):
        tokens = [token.lower() for token in tokens]
        vectors = [self.glove_vectors[token].numpy() if token in self.glove_vectors else self.glove_vectors["unk"].numpy() for token in tokens]
        return vectors

    def pad_sequence(self, sequence, padding_value):
        if len(sequence) > self.max_seq_len:
            return sequence[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(sequence)
            return sequence + [padding_value] * pad_len

    def create_samples(self):
        samples = []
        for relation, instances in self.data.items():
            for instance in instances:
                tokens = instance["tokens"]
                head_pos = instance["h"][2][0][0]
                tail_pos = instance["t"][2][0][0]

                token_vectors = self.tokenize_and_vectorize(tokens)
                token_vectors = self.pad_sequence(token_vectors, self.glove_vectors["unk"].numpy())

                mask = [1 if i < head_pos else 2 if i < tail_pos else 3 for i in range(len(tokens))]
                mask = self.pad_sequence(mask, 0)  # Use 0 for padding the mask

                samples.append({
                    "vectors": token_vectors,
                    "mask": mask,
                    "relation": relation
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vectors = torch.tensor(sample["vectors"], dtype=torch.float)
        mask = torch.tensor(sample["mask"], dtype=torch.long)
        label = self.label_encoder.transform([sample["relation"]])[0]
        return vectors, mask, label

def fewrel_dataloader(data_path):
    
    # Load GloVe embeddings using torchtext
    glove_vectors = Vectors(name='glove.6B.50d.txt', cache='embeddings/')

    dataset = FewRelDataset(data_path, glove_vectors)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Testing the DataLoader
    test_dataloader(dataloader)

    return dataloader

def test_dataloader(dataloader):
    for i, batch in enumerate(dataloader):
        vectors, mask, labels = batch
        print(f"Batch {i+1}")
        print("Vectors shape:", vectors.shape)
        print("Mask shape:", mask.shape)
        print("Labels shape:", labels.shape)
        print("Sample Vectors:", vectors[0])
        print("Sample Mask:", mask[0])
        print("Sample Label:", labels[0])
        if i == 2:  # Print the first 3 batches for inspection
            break