#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  train_localtest.py
@Time    :  :2023/01/19
@Author  :   Dr. Cat Lu / BFcat
@Version :   1.0
@Contact :   bfcat@live.cn
@Site    :   https://bg4xsd.github.io
@License :   (C)MIT License
@Desc    :   This is a part of project CWLab, more details can be found on the site.
'''

import os
import sys

os.chdir(sys.path[0])
# print("Current work directory -> %s" % os.getcwd())

import random
import argparse

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import wandb


from morse import ALPHABET, generate_sample
from decoder import prediction_to_str, collate_fn_pad, NetDLD, Dataset


num_tags = len(ALPHABET)

# Create directory, 创建目录
def Mkdir(path):  # path是指定文件夹路径
    if os.path.isdir(path):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(path)


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
    # usage: python train.py --lr=0.01 --epoch_start 2500 --epoch_end 7500
    # usage: python train.py  --len_low 12 --len_high 24 --batch_size 64 --lr 0.001 --epoch_start 0 --epoch_end 2000 --workers 6 --output_dir models
    #        python train.py  --len_low 1 --len_high 2 --batch_size 32 --lr 0.001 --epoch_start 0 --epoch_end 2000 --workers 6 --output_dir models
    # usage: python train.py  --len_low 12 --len_high 24 --batch_size 64 --lr 0.0001 --epoch_start 8000 --epoch_end 10000 --workers 6 --output_dir models
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
            50  # default is  training from 0~50 epoch， try and see the case
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

    myConfig = {
        "architecture": "DLD",  # Original neural network structure
        "dataset": "Synthetic-0.7-2.0",  # Noise scale 0.7~2.0
        "learning_rate": args.lr,
        "len_low": args.len_low,
        "len_high": args.len_high,
        "batch_size": args.batch_size,
        "epoch_start": args.epoch_start,
        "epoch_end": args.epoch_end,
        "workers": args.workers,
        "dicts": 59,
        "output_dir": args.output_dir,
    }

    myConfigFix = {
        "architecture": "DLD",  # Original neural network structure
        "dataset": "Synthetic-0.7-2.0",  # Noise scale 0.7~2.0
        "learning_rate": 0.001,
        "len_low": 10,
        "len_high": 30,
        "batch_size": 64,
        "epoch_start": 0,
        "epoch_end": 10,
        "workers": 2,
        "dicts": 59,
        "output_dir": "./models",
    }

    # While debug, the fixxed params can be used
    myConfig = myConfigFix

    batch_size = int(myConfig['batch_size'])
    workers = int(myConfig['workers'])
    LR = float(myConfig['learning_rate'])
    len_low = int(myConfig['len_low'])
    len_high = int(myConfig['len_high'])
    epoch_start = int(myConfig['epoch_start'])
    epoch_end = int(myConfig['epoch_end'])
    output_dir = str(myConfig['output_dir'])

    model_dir = output_dir
    if not os.path.isdir(model_dir):
        Mkdir(model_dir)

    # 获得参数大小，用来构建网络， 这个size 跟抽样频率有关
    spectrogram_size = generate_sample()[1].shape[0]

    # Get cpu or gpu device for training.
    mydevice = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {mydevice} device")
    # device = torch.device(mydevice)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    writer = SummaryWriter()

    # Set up trainer & evaluator
    model = NetDLD(num_tags, spectrogram_size).to(device)
    print("Number of params", model.count_parameters())

    # Lower learning rate to 1e-4 after about 1500 epochs
    optimizer = optim.Adam(model.parameters(), lr=float(LR))
    # optimizer = optim.Adam(model.parameters(), lr=1e-3 ~ 1e-4)
    ctc_loss = nn.CTCLoss()

    train_loader = torch.utils.data.DataLoader(
        # Dataset(), # Default init using fix length
        Dataset(int(len_low), int(len_high)),  # Imnproved init using choosing length
        batch_size=int(batch_size),
        num_workers=int(workers),
        collate_fn=collate_fn_pad,
    )

    random.seed(0)

    # epoch = 1500 # modify with lr=1e-4
    epoch = int(epoch_start)

    # Resume training
    if epoch != 0:
        model.load_state_dict(
            torch.load(model_dir + f"/{epoch:06}.pt", map_location=device)
        )

    model.train()
    while epoch <= int(epoch_end):
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

        if epoch % 50 == 0:
            torch.save(model.state_dict(), model_dir + f"/{epoch:06}.pt")

        print(prediction_to_str(y[0]))
        print(prediction_to_str(m))
        print(loss.item())
        print()
        epoch += 1
