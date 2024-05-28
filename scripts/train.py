import os
import json

import numpy as np

from sentence_encoder import CNNSentenceEncoder, test_sentence_encoder
from data_loader import get_loader, test_data_loader

def main():

    model_name = "proto"
    encoder_name = "cnn"
    trainN = 5 # N in N-way K-shot. trainN is the specific N in training process.
    N = 5 # N in N-way K-shot.
    K = 1 # K in N-way K-shot.
    Q = 1 #Sample Q query instances for each relation.
    batch_size = 4 # default is 4 as per the paper
    max_length = 128 # default is 128 as per the paper
    val_step = 1000
    na_rate = 0 # NA rate for FewRel 2.0 none-of-the-above (NOTA) detection. Note that here na_rate specifies the rate
                # between Q for NOTA and Q for positive. For example, na_rate=0 means the normal setting, na_rate=1,2,5
                # corresponds to NA rate = 15%, 30% and 50% in 5-way settings.
    train = "train_wiki"
    val = "val_wiki"

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    try:
        glove_mat = np.load('./pretrain/glove_mat.npy')
        glove_word2id = json.load(open('./pretrain/glove_word2id.json'))
        print("GloVe embeddings loaded successfully.")
        print("GloVe matrix shape:", glove_mat.shape)
        print("Number of words in GloVe dictionary:", len(glove_word2id))
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
    
    sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length)
    print("Testing sentence encoder")
    # test_sentence_encoder(sentence_encoder)
    print("CNNSentenceEncoder instantiated successfully.")

    train_data_loader = get_loader(train, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=na_rate, batch_size=batch_size)
    val_data_loader = get_loader(val, sentence_encoder, N=N, K=K, Q=Q, na_rate=na_rate, batch_size=batch_size)
    # Test data loaders
    # print("Testing train data loader")
    # test_data_loader(train_data_loader)
    # print("Testing val data loader")
    # test_data_loader(val_data_loader)
    print("Train and Test DataLoader instantiated successfully.")



if __name__ == "__main__":
    main()