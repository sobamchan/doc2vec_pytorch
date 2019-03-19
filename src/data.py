from pathlib import Path
import os.path as osp
import pickle
from collections import Counter

import fire
import spacy
import lineflow as lf


NLP = spacy.load('en_core_web_sm')
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'


def preprocess(sent):
    instance = [
        token.text.lower()
        for token in NLP(sent)
        if not token.is_space
    ]
    return instance


def build_vocab(tokens, cache='vocab.pkl', max_size=500000):
    if not osp.isfile(cache):
        counter = Counter(tokens)
        words, _ = zip(*counter.most_common(max_size))
        words = [PAD_TOKEN, UNK_TOKEN] + list(words)
        t2i = dict(zip(words, range(len(words))))
        if START_TOKEN not in t2i:
            t2i[START_TOKEN] = len(t2i)
            words += [START_TOKEN]
        if END_TOKEN not in t2i:
            t2i[END_TOKEN] = len(t2i)
            words += [END_TOKEN]
        with open(cache, 'wb') as f:
            pickle.dump((t2i, words), f)
    else:
        with open(cache, 'rb') as f:
            t2i, words = pickle.load(f)
    return t2i, words


def postprocess(t2i, unk_index):
    def _f(x):
        return {
            'id': x[0],
            'tokens': [t2i.get(token, unk_index) for token in x[1]]
        }
    return _f


def build(datapath='./data/example.txt', savedir='./'):
    datapath = Path(datapath)
    savedir = Path(savedir)

    docs = lf.TextDataset(str(datapath))
    ids = lf.Dataset(range(len(docs)))
    docs = docs.map(preprocess)
    ds = lf.zip(docs, ids)

    tokens = lf.flat_map(
        lambda x: x[1],
        ds,
        lazy=True
    )
    t2i, words = build_vocab(tokens, str(savedir / 'vocab.pkl'))

    unk_index = t2i[UNK_TOKEN]

    ds.map(postprocess(t2i, unk_index))\
        .save(str(savedir / 'dataset.token.pkl'))


if __name__ == '__main__':
    fire.Fire()
