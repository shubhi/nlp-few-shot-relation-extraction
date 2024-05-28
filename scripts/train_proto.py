import os
import json

import numpy as np

import torch
from torch import optim

from util.sentence_encoder import CNNSentenceEncoder, sentence_encoder_verify
from util.data_loader import get_loader, data_loader_verify
from util.framework import FewShotREFramework
from models.proto import Proto

def main():

    model_name = "proto"
    encoder_name = "cnn"
    trainN = 5 # N in N-way K-shot. trainN is the specific N in training process.
    N = 5 # N in N-way K-shot.
    K = 1 # K in N-way K-shot.
    Q = 1 #Sample Q query instances for each relation.
    batch_size = 4 # default is 4 as per the paper
    max_length = 128 # default is 128 as per the paper
    na_rate = 0 # NA rate for FewRel 2.0 none-of-the-above (NOTA) detection. Note that here na_rate specifies the rate
                # between Q for NOTA and Q for positive. For example, na_rate=0 means the normal setting, na_rate=1,2,5
                # corresponds to NA rate = 15%, 30% and 50% in 5-way settings.
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    try:
        glove_mat = np.load('./pretrain/glove_mat.npy')
        glove_word2id = json.load(open('./pretrain/glove_word2id.json'))
        print("GloVe embeddings loaded successfully.")

        # Add entries for [UNK] and [PAD] tokens in the GloVe matrix
        unk_embedding = np.random.randn(1, glove_mat.shape[1])  # Random initialization for [UNK]
        pad_embedding = np.zeros((1, glove_mat.shape[1]))       # Zero initialization for [PAD]

        # Extend the GloVe matrix
        glove_mat = np.vstack([glove_mat, unk_embedding, pad_embedding])

        # Update glove_word2id dictionary to include [UNK] and [PAD]
        glove_word2id['[UNK]'] = glove_mat.shape[0] - 2
        glove_word2id['[PAD]'] = glove_mat.shape[0] - 1

        # Convert GloVe matrix to NumPy array if it's a tensor
        if isinstance(glove_mat, torch.Tensor):
            glove_mat = glove_mat.numpy()

        print(f"Extended GloVe matrix shape: {glove_mat.shape}")
        print(f"Number of words in GloVe dictionary: {len(glove_word2id)}")
    except:
        raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
    
    try:
        sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length)
        # print("Verify sentence encoder")
        # sentence_encoder_verify(sentence_encoder)
        print("CNNSentenceEncoder instantiated successfully.")
    except:
        raise Exception("CNN Sentence Encoder failed.")

    
    train = "train_wiki"
    val = "val_wiki"
    try:
        train_data_loader = get_loader(train, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=na_rate, batch_size=batch_size)
        val_data_loader = get_loader(val, sentence_encoder, N=N, K=K, Q=Q, na_rate=na_rate, batch_size=batch_size)
        # Verify data loaders
        # print("Testing train data loader")
        # data_loader_verify(train_data_loader)
        # print("Testing val data loader")
        # data_loader_verify(val_data_loader)
        print("Train and Test DataLoader instantiated successfully.")
    except:
        raise Exception("Data Loader generation failed.")
    

    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA.")
    else:
        print("Using CPU")
    
    dot = False # use dot instead of L2 distance for proto
    save_ckpt = None
    lr = 1e-1 # Default is -1
    pytorch_optim = optim.Adam
    train_iter = 30000 # num of iters in training
    val_iter = 1000 # num of iters in validation
    grad_iter = 1 # accumulate gradient every x iterations
    val_step = 2000 # val after training how many iters
    load_ckpt = None
    ckpt_name = ""

    prefix = '-'.join([model_name, encoder_name, train, val, str(N), str(K)])
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
        
    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=dot)
    else:
        raise NotImplementedError

    try:  
        framework = FewShotREFramework(train_data_loader, val_data_loader)
        print("FewShortREFFramework setup successfully.")
    except:
        raise Exception("FewShortREFFramework setup failed.")

    
    train_iter = train_iter * grad_iter
    framework.train(model, prefix, batch_size, trainN, N, K, Q,
            pytorch_optim=pytorch_optim, load_ckpt=load_ckpt, save_ckpt=ckpt,
            na_rate=na_rate, val_step=val_step,
            train_iter=train_iter, val_iter=val_iter,
            learning_rate=lr, grad_iter=grad_iter)

    # acc = framework.eval(model, batch_size, N, K, Q, test_iter, na_rate=na_rate, ckpt=ckpt)
    # print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()