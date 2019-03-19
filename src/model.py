import torch
import torch.nn as nn


class PVDM(nn.Module):

    def __init__(self, voc_n, doc_n, hid_n):
        super().__init__()

        self.D = nn.Parameter(
                torch.randn(doc_n, hid_n)
                )
        self.W = nn.Parameter(
                torch.randn(voc_n, hid_n)
                )

        self.Out = nn.Parameter(
                torch.FloatTensor(1, hid_n, voc_n).zero_()
                )

    def forward(self, doc_ids, contexts):
        bsize = len(doc_ids)

        # x -> [B, H]
        x = torch.add(
                self.D[doc_ids, :],
                torch.sum(self.W[contexts, :], dim=1)
                )

        return torch.bmm(
                x.unsqueeze(1),
                self.Out.repeat(bsize, 1, 1)
                ).squeeze(1)
