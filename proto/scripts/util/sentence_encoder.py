import torch
import torch.nn as nn
import numpy as np
import logging
from util.embedding import Embedding
from util.encoder import Encoder


# Configure logging
logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, filemode='w')

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
                 pos_embedding_dim=5, hidden_size=230):
        super(CNNSentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word2id = word2id

        self.embedding = Embedding(word_vec_mat, max_length, 
                                   word_embedding_dim, pos_embedding_dim)
        self.encoder = Encoder(max_length, word_embedding_dim, 
                               pos_embedding_dim, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.pad_token = self.word2id['[PAD]']
        self.unk_token = self.word2id['[UNK]']

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x = self.encoder(x)
        return x

    def tokenize(self, tokens, pos1, pos2):
        indexed_tokens = [self.word2id.get(w, self.word2id['[UNK]']) for w in tokens]
        indexed_pos1 = [p + self.max_length for p in pos1]
        indexed_pos2 = [p + self.max_length for p in pos2]
        if len(indexed_tokens) < self.max_length:
            indexed_tokens += [self.word2id['[PAD]']] * (self.max_length - len(indexed_tokens))
            indexed_pos1 += [self.word2id['[PAD]']] * (self.max_length - len(indexed_pos1))
            indexed_pos2 += [self.word2id['[PAD]']] * (self.max_length - len(indexed_pos2))
        indexed_tokens = indexed_tokens[:self.max_length]
        indexed_pos1 = indexed_pos1[:self.max_length]
        indexed_pos2 = indexed_pos2[:self.max_length]
        
        # Debugging info
        max_word_index = max(indexed_tokens)
        logging.debug(f"Max word index: {max_word_index}")
        if max_word_index >= len(self.word2id):
            logging.error(f"Word index out of range: {max_word_index} >= {len(self.word2id)}")
            raise ValueError(f"Word index out of range: {max_word_index} >= {len(self.word2id)}")
        
        mask = [1 if token != self.word2id['[PAD]'] else 0 for token in indexed_tokens]
        
        return indexed_tokens, indexed_pos1, indexed_pos2, mask



def sentence_encoder_verify(sentence_encoder):
    # Create dummy data
    raw_tokens = ["This", "is", "a", "test", "sentence", "."]
    pos_head = [1, 1]
    pos_tail = [4, 4]
    
    indexed_tokens, pos1, pos2, mask = sentence_encoder.tokenize(raw_tokens, pos_head, pos_tail)
    
    print("Indexed tokens before clipping:", indexed_tokens)
    
    word_vec_size = sentence_encoder.embedding.word_embedding.weight.size(0)
    indexed_tokens = [min(token, word_vec_size - 1) for token in indexed_tokens]
    
    print("Indexed tokens after clipping:", indexed_tokens)
    print("Position 1:", pos1)
    print("Position 2:", pos2)
    print("Mask:", mask)
    
    max_word_index = max(indexed_tokens)
    max_pos1_index = max(pos1)
    max_pos2_index = max(pos2)
    print("Max word index:", max_word_index)
    print("Max pos1 index:", max_pos1_index)
    print("Max pos2 index:", max_pos2_index)
    
    pos1_vec_size = sentence_encoder.embedding.pos1_embedding.weight.size(0)
    pos2_vec_size = sentence_encoder.embedding.pos2_embedding.weight.size(0)
    print("Word embedding size:", word_vec_size)
    print("Pos1 embedding size:", pos1_vec_size)
    print("Pos2 embedding size:", pos2_vec_size)
    
    assert max_word_index < word_vec_size, "Word index out of range"
    assert max_pos1_index < pos1_vec_size, "Pos1 index out of range"
    assert max_pos2_index < pos2_vec_size, "Pos2 index out of range"
    
    inputs = {
        'word': torch.tensor(indexed_tokens).unsqueeze(0),  
        'pos1': torch.tensor(pos1).unsqueeze(0),           
        'pos2': torch.tensor(pos2).unsqueeze(0),            
        'mask': torch.tensor(mask).unsqueeze(0)             
    }
    
    with torch.no_grad():
        output = sentence_encoder(inputs)
    
    print("Output shape:", output.shape)
    print("Output type:", type(output))