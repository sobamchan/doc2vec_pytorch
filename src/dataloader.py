import random

import torch
import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler


# def collate_fn(batch):
#     for sample in batch:
#         for k, v in sample.items():
#             sample[k] = torch.tensor(v)
#     return batch


def collate_fn(batch):
    doc_ids = []
    contexts = []
    targets = []

    for sample in batch:
        doc_ids.append(sample['doc_id'])
        contexts.append(sample['context'])
        targets.append(sample['target'])
    return (
            torch.LongTensor(doc_ids),
            torch.LongTensor(contexts),
            torch.LongTensor(targets)
            )


class Dataset(data.Dataset):

    def __init__(self, ds, context_size, pad_index):
        self.ds = ds  # [{id: int, tokens: list}...]
        self.context_size = context_size
        self.pad_index = pad_index

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]
        doc_id, tokens = data['id'], data['tokens']
        ntokens = len(tokens)
        pad_index = self.pad_index
        tokens =\
            [pad_index] * self.context_size +\
            tokens +\
            [pad_index] * self.context_size

        sidx = random.randint(0, ntokens)
        eidx = sidx + self.context_size
        target = tokens[eidx+1]

        return {
                'doc_id': doc_id,
                'context': tokens[sidx:eidx],
                'target': target
                }


def get_loaders(ds, context_size, pad_index, bsize):
    dataset = Dataset(ds, context_size, pad_index)
    return data.DataLoader(
            dataset,
            batch_size=bsize,
            sampler=RandomSampler(dataset),
            collate_fn=collate_fn
            )
