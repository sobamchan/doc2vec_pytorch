from pathlib import Path
import pickle

import numpy as np
import fire
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import PAD_TOKEN
from src.dataloader import get_loaders
from src.model import PVDM


def run(datadir, savedir, context_size=4, bsize=32, hid_n=300, lr=0.001,
        epoch=10, use_cuda=True, use_tb=True):

    datadir = Path(datadir)
    if use_tb:
        tb = SummaryWriter(savedir)

    t2i, words = pickle.load(open(datadir / 'vocab.pkl', 'rb'))
    dataset = pickle.load(open(datadir / 'dataset.token.pkl', 'rb'))

    dataloader = get_loaders(dataset, context_size, t2i[PAD_TOKEN], bsize)

    voc_n = len(t2i)
    doc_n = len(dataset)

    device = torch.device('cuda' if use_cuda else 'cpu')

    print('Voc n:', voc_n)
    print('Doc n:', doc_n)
    print('Device: ', device)

    model = PVDM(voc_n, doc_n, hid_n)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()
    best_loss = 1e+5

    for i_epoch in range(1, epoch + 1):
        losses = []
        for doc_ids, contexts, targets in dataloader:
            # doc_ids -> [B]
            # contexts -> [B, context_size]
            # targets -> [B]
            doc_ids = doc_ids.to(device)
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            y = model(doc_ids, contexts)  # [B, V]
            loss = loss_func(y, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        mean_loss = np.mean(losses)
        print('Loss: ', mean_loss)

        if use_tb:
            tb.add_scalar(
                    'train/loss', mean_loss, i_epoch
                    )

        if mean_loss < best_loss:
            best_loss = mean_loss
            # Dump model
            model.cpu()
            with open(savedir / 'D.pth', 'wb') as f:
                pickle.dump(model.D, f)
            model.to(device)


if __name__ == '__main__':
    fire.Fire()
