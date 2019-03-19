from pathlib import Path
import pickle

import fire

from src.data import PAD_TOKEN
from src.dataloader import get_loaders


def run(datadir, context_size=4, bsize=32):
    datadir = Path(datadir)

    t2i, words = pickle.load(open(datadir / 'vocab.pkl', 'rb'))
    dataset = pickle.load(open(datadir / 'dataset.token.pkl', 'rb'))

    dataloader = get_loaders(dataset, context_size, t2i[PAD_TOKEN], bsize)

    for batch in dataloader:
        print(batch)


if __name__ == '__main__':
    fire.Fire()
