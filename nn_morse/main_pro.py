#!/usr/bin/env python3
import os
import sys

os.chdir(sys.path[0])
print("Current work directory -> %s" % os.getcwd())

import random
import argparse

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


from morse import ALPHABET, generate_sample
from itertools import groupby

num_tags = len(ALPHABET)

# 0: blank label
tag_to_idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
idx_to_tag = {i + 1: c for i, c in enumerate(ALPHABET)}

# Create directory, 创建目录
def Mkdir(path):  # path是指定文件夹路径
    if os.path.isdir(path):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(path)


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


class Net(nn.Module):
    def __init__(self, num_tags, spectrogram_size):
        super(Net, self).__init__()
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


if __name__ == "__main__":
    # Add new input args for running batch jobs
    # len_high must > 1,  the proc use random.randrange(a,b)
    # In fact,it will generate data from a to b-1.
    parser = argparse.ArgumentParser()
    parser.add_argument("--len_low")
    parser.add_argument("--len_high")
    parser.add_argument("--batch_size")
    parser.add_argument("--lr", required=True)
    parser.add_argument("--epoch_start", required=True)
    parser.add_argument("--epoch_end", required=True)
    parser.add_argument("--workers")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    # usage: python main_pro.py --lr=0.01 --epoch_start 2500 --epoch_end 7500
    # usage: python main_pro.py  --len_low 12 --len_high 24 --batch_size 64 --lr 0.001 --epoch_start 0 --epoch_end 2000 --workers 6 --output_dir models
    #        python main_pro.py  --len_low 1 --len_high 2 --batch_size 32 --lr 0.001 --epoch_start 0 --epoch_end 2000 --workers 6 --output_dir models
    # usage: python main_pro.py  --len_low 12 --len_high 24 --batch_size 64 --lr 0.0001 --epoch_start 8000 --epoch_end 10000 --workers 6 --output_dir models
    # if not pointed the training sentence, use default value, 10-20 chars
    if args.len_low is None:
        args.len_low = 10
    if args.len_high is None:
        args.len_high = 20
    if args.batch_size is None:
        args.batch_size = 64  # default is 64, sometimes, 128 will be OK.
    if args.lr is None:
        args.lr = 1e-3  # default is 0.001, after many epoch,it should be reduce to 1e-4 or less
    if args.epoch_start is None:
        args.epoch_start = 0
    if args.epoch_end is None:
        args.epoch_end = (
            5000  # default is  training from 0~5000 epoch， try and see the case
        )
    if args.workers is None:
        args.workers = 4  # use 4 for lower cpu
    if args.output_dir is None:
        args.output_dir = "models"  # Model output directory
    print(
        "QSO length is ",
        args.len_low,
        " ~ ",
        int(args.len_high) - 1,
        ", traing batch size is ",
        args.batch_size,
        ", LR is ",
        args.lr,
        ", Starting epoch is ",
        args.epoch_start,
        ", Ending epoch is ",
        args.epoch_end,
        ", Works' number is ",
        args.workers,
        ", Model output dir is ",
        args.output_dir,
    )

    model_dir = args.output_dir
    if not os.path.isdir(model_dir):
        Mkdir(model_dir)

    # 获得参数大小，用来构建网络， 这个size 跟抽样频率有关
    spectrogram_size = generate_sample()[1].shape[0]

    device = torch.device("cuda")
    # device = torch.device("cpu")
    writer = SummaryWriter()

    # Set up trainer & evaluator
    model = Net(num_tags, spectrogram_size).to(device)
    print("Number of params", model.count_parameters())

    # Lower learning rate to 1e-4 after about 1500 epochs
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    # optimizer = optim.Adam(model.parameters(), lr=1e-3 ~ 1e-4)
    ctc_loss = nn.CTCLoss()

    train_loader = torch.utils.data.DataLoader(
        # Dataset(), # Default init using fix length
        Dataset(
            int(args.len_low), int(args.len_high)
        ),  # Imnproved init using choosing length
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
        collate_fn=collate_fn_pad,
    )

    random.seed(0)

    # epoch = 1500 # modify with lr=1e-4
    epoch = int(args.epoch_start)

    # Resume training
    if epoch != 0:
        model.load_state_dict(
            torch.load(model_dir + f"/{epoch:06}.pt", map_location=device)
        )

    model.train()
    while epoch <= int(args.epoch_end):
        # if epoch % 200 == 0:   # every 1500 epoch, update lr rate
        #     for params in optimizer.param_groups:
        #         # Find in the params list，update the lr = lr * 0.9
        #         params['lr'] *= 0.
        #         # params['weight_decay'] = 0.5  # Others
        for (input_lengths, output_lengths, x, y) in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            m = torch.argmax(y_pred[0], 1)
            y_pred = y_pred.permute(1, 0, 2)

            loss = ctc_loss(y_pred, y, input_lengths, output_lengths)

            loss.backward()
            optimizer.step()

        writer.add_scalar("training/loss", loss.item(), epoch)

        if epoch % 200 == 0:
            torch.save(model.state_dict(), model_dir + f"/{epoch:06}.pt")

        print(prediction_to_str(y[0]))
        print(prediction_to_str(m))
        print(loss.item())
        print()
        epoch += 1
