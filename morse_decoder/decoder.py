#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  morse.py
@Time    :  :2022/12/10
@Author  :   Dr. Cat Lu / BFcat
@Version :   1.0
@Contact :   bfcat@live.cn
@Site    :   https://bg4xsd.github.io
@License :   (C)MIT License
@Desc    :   This is a part of project CWLab, more details can be found on the site.
'''

import os, sys

os.chdir(sys.path[0])
# print("Current work directory -> %s" % os.getcwd())

from scipy import signal
import numpy as np
import random
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from itertools import groupby

from morse import ALPHABET, generate_sample

# 0: blank label
tag_to_idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
idx_to_tag = {i + 1: c for i, c in enumerate(ALPHABET)}

num_tags = len(ALPHABET)


def prediction_to_str(seq):
    if not isinstance(seq, list):
        seq = seq.tolist()

    # remove duplicates
    seq = [i[0] for i in groupby(seq)]

    # remove blanks
    seq = [s for s in seq if s != 0]

    # convert to string
    seq = "".join(idx_to_tag[c] for c in seq)

    return seq


def get_training_sample(*args, **kwargs):
    _, spec, y = generate_sample(*args, **kwargs)
    spec = torch.from_numpy(spec)
    spec = spec.permute(1, 0)

    y_tags = [tag_to_idx[c] for c in y]
    y_tags = torch.tensor(y_tags)

    return spec, y_tags


# It is a Dense-LSTM_Dense network by PyTorch
class NetDLD(nn.Module):
    def __init__(self, num_tags, spectrogram_size):
        super(NetDLD, self).__init__()
        # Here add ONE more, is say for 0, stands for blank
        # But why not the last tag, 60?
        num_tags = num_tags + 1  # 0: blank
        hidden_dim = 256
        lstm_dim1 = 256

        self.dense1 = nn.Linear(spectrogram_size, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense4 = nn.Linear(hidden_dim, lstm_dim1)
        self.lstm1 = nn.LSTM(lstm_dim1, lstm_dim1, batch_first=True)
        self.dense5 = nn.Linear(lstm_dim1, num_tags)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        x, _ = self.lstm1(x)

        x = self.dense5(x)
        x = F.log_softmax(x, dim=2)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Dataset(data.Dataset):
    # Defualt QSO length, low limit and upper limit
    low = 10
    high = 20

    def __init__(self, inLow, inHigh):
        self.low = inLow
        self.high = inHigh

    def __len__(self):
        return 2048

    def __getitem__(self, index):
        # Modify here, for input cw length, default recomends is  10~20,
        # For real audio data, buffer is short and use 1~10(not include 10, only 1~9)
        # randrange(1,3), only will be 1, 2, NOT include 3,
        # so the input should use lenght+1
        length = random.randrange(self.low, self.high)
        pitch = random.randrange(100, 950)
        wpm = random.randrange(10, 40)
        noise_power = random.randrange(0, 200)
        amplitude = random.randrange(10, 150)
        return get_training_sample(length, pitch, wpm, noise_power, amplitude)


def collate_fn_pad(batch):
    xs, ys = zip(*batch)

    input_lengths = torch.tensor([t.shape[0] for t in xs])
    output_lengths = torch.tensor([t.shape[0] for t in ys])

    seqs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)

    return input_lengths, output_lengths, seqs, ys
