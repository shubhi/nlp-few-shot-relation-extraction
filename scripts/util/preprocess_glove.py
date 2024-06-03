# scripts/preprocess_glove.py

import numpy as np
import json
import os

def preprocess_glove(glove_file, output_dir):
    word2id = {}
    embeddings = []

    with open(glove_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            word2id[word] = idx
            embeddings.append(embedding)

    embeddings = np.array(embeddings)
    np.save(os.path.join(output_dir, "glove_mat.npy"), embeddings)
    with open(os.path.join(output_dir, "glove_word2id.json"), 'w') as f:
        json.dump(word2id, f)

if __name__ == "__main__":
    glove_file = 'pretrain/glove.6B.300d.txt'
    output_dir = 'pretrain'
    preprocess_glove(glove_file, output_dir)
