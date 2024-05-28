import torch
import torch.nn as nn
import logging

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5):
        super(Embedding, self).__init__()

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        
        # Word embedding
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']

        # Validate indices
        logging.debug(f"Max word index: {torch.max(word).item()}")
        logging.debug(f"Max pos1 index: {torch.max(pos1).item()}")
        logging.debug(f"Max pos2 index: {torch.max(pos2).item()}")

        if torch.max(word) >= self.word_embedding.num_embeddings:
            raise ValueError(f"Word index out of range: {torch.max(word).item()} >= {self.word_embedding.num_embeddings}")
        if torch.max(pos1) >= self.pos1_embedding.num_embeddings:
            raise ValueError(f"Position 1 index out of range: {torch.max(pos1).item()} >= {self.pos1_embedding.num_embeddings}")
        if torch.max(pos2) >= self.pos2_embedding.num_embeddings:
            raise ValueError(f"Position 2 index out of range: {torch.max(pos2).item()} >= {self.pos2_embedding.num_embeddings}")

        x = torch.cat([self.word_embedding(word), self.pos1_embedding(pos1), self.pos2_embedding(pos2)], 2)
        return x
