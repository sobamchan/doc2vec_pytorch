from pathlib import Path
import pickle

import fire
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def most_similar(datadir, textpath, modelpath, doc_id, topn=5):
    datadir = Path(datadir)
    textpath = Path(textpath)
    modelpath = Path(modelpath)

    docs = open(textpath, 'r').readlines()

    print('TARGET DOCUMENT: id -> %d, text -> %s' % (doc_id, docs[doc_id]))

    D = pickle.load(open(modelpath, 'rb')).data.numpy()

    base_vec = np.expand_dims(D[doc_id], axis=0)

    sims = np.argsort(cosine_similarity(base_vec, D)[0])[::-1]

    print(np.array(docs)[sims[:topn]])


if __name__ == '__main__':
    fire.Fire()
