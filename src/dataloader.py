import pickle

import torch
import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler


def collate_fn(batch):
    for sample in batch:
        sample['id'] = torch.tensor(sample['id'])
        sample['tokens'] = torch.tensor(sample['tokens'])
    return batch


class Dataset(data.Dataset):

    def __init__(self, ds):
        self.ds = ds  # [{id: int, tokens: list}...]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return {
                'id': self.ds[idx]['id'],
                'tokens': self.ds[idx]['tokens'],
                }


def get_loaders(dataset_path, bsize):
    ds = pickle.load(open(dataset_path, 'rb'))

    dataset = Dataset(ds)
    return data.DataLoader(
            dataset,
            batch_size=bsize,
            sampler=RandomSampler(dataset),
            collate_fn=collate_fn
            )
