import os
import json

import numpy as np

from sentence_encoder import *

def main():

    data_directory = "data"
    train_path = os.path.join(data_directory,"train_wiki.json")
    val_path = os.path.join(data_directory,"val_pubmed.json")

    model_name = "proto"
    encoder_name = "cnn"
    trainN = 5 # N in N-way K-shot. trainN is the specific N in training process.
    N = 5 # N in N-way K-shot.
    K = 1 # K in N-way K-shot.
    Q = 1 #Sample Q query instances for each relation.
    batch_size = 4 # default is 4 as per the paper
    max_length = 128 # default is 128 as per the paper
    val_step = 1000

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
    # test_sentence_encoder(sentence_encoder)
    print("CNNSentenceEncoder instantiated successfully.")
    




if __name__ == "__main__":
    main()