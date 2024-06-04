# scripts/train_maml.py

import os
import json
import numpy as np
import torch
from torch import optim
from util.sentence_encoder import CNNSentenceEncoder
from util.data_loader import get_loader
from util.framework import FewShotREFramework
from models.maml import MAML

def main():
    model_name = "maml"
    encoder_name = "cnn"
    trainN = 5
    N = 5
    K = 1
    Q = 1
    batch_size = 4
    max_length = 16
    na_rate = 0

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    try:
        glove_mat = np.load('./pretrain/glove_mat.npy')
        glove_word2id = json.load(open('./pretrain/glove_word2id.json'))
        print("GloVe embeddings loaded successfully.")

        unk_embedding = np.random.randn(1, glove_mat.shape[1])
        pad_embedding = np.zeros((1, glove_mat.shape[1]))

        glove_mat = np.vstack([glove_mat, unk_embedding, pad_embedding])
        glove_word2id['[UNK]'] = glove_mat.shape[0] - 2
        glove_word2id['[PAD]'] = glove_mat.shape[0] - 1

        # Check the shape of the GloVe matrix
        word_embedding_dim = glove_mat.shape[1]

    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return

    try:
        train_data_loader = get_loader('train_wiki', encoder_name, N, K, Q, batch_size, max_length, glove_mat, glove_word2id)
        val_data_loader = get_loader('val_wiki', encoder_name, N, K, Q, batch_size, max_length, glove_mat, glove_word2id)
        print("Train and Test DataLoader instantiated successfully.")
    except Exception as e:
        print(f"Error generating DataLoader: {e}")
        return

    try:
        sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length, word_embedding_dim=word_embedding_dim)
        print("Sentence encoder instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating sentence encoder: {e}")
        return

    if torch.cuda.is_available():
        sentence_encoder.cuda()
        print("Using CUDA.")
    else:
        print("Using CPU")

    try:
        model = MAML(sentence_encoder, inner_lr=0.01, num_inner_steps=5)
        print("MAML model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating MAML model: {e}")
        return

    dot = False
    save_ckpt = None
    lr = 1e-3
    pytorch_optim = optim.Adam
    train_iter = 30000
    val_iter = 1000
    grad_iter = 1
    val_step = 2000
    load_ckpt = None
    ckpt_name = ""

    prefix = '-'.join([model_name, encoder_name, str(N), str(K)])
    if na_rate != 0:
        prefix += '-na{}'.format(na_rate)
    if dot:
        prefix += '-dot'
    if len(ckpt_name) > 0:
        prefix += '-' + ckpt_name

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if save_ckpt:
        ckpt = save_ckpt

    try:
        outer_optimizer = pytorch_optim(model.parameters(), lr=lr)
        print("Optimizer instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating optimizer: {e}")
        return

    train_iter = train_iter * grad_iter
    for iteration in range(train_iter):
        try:
            support_set, query_set = next(train_data_loader)
            query_loss, accuracy = model(support_set, query_set, N, K, Q)
            model.meta_update(outer_optimizer, query_loss)

            if (iteration + 1) % val_step == 0:
                val_support_set, val_query_set = next(val_data_loader)
                val_query_loss, val_accuracy = model(val_support_set, val_query_set, N, K, Q)
                print(f'Validation Loss: {val_query_loss.item()}, Validation Accuracy: {val_accuracy}')

            if (iteration + 1) % 1000 == 0:
                torch.save(model.state_dict(), ckpt)
                print(f'Saved checkpoint at iteration {iteration + 1}')
        except Exception as e:
            print(f"Error during training iteration {iteration + 1}: {e}")
            break

if __name__ == "__main__":
    main()
