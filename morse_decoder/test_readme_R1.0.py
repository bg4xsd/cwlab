#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  test_readme_R1.0.py
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
print("Current work directory -> %s" % os.getcwd())

import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import torch
from scipy import signal

from decoder import NetDLD, num_tags, prediction_to_str
from morse import ALPHABET, SAMPLE_FREQ, get_spectrogram

# For example
# python decode_audio.py --model ./models/001750.pt ../CallCQ_pitch416_wpm17_noise83_amplitude36.wav


def decoder_realtime(modelfile, data):
    print("TODO")


def decoder_file(modelfile, wavfile):
    rate, data = scipy.io.wavfile.read(wavfile)

    # Resample and rescale
    length = len(data) / rate
    new_length = int(length * SAMPLE_FREQ)

    data = signal.resample(data, new_length)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # Create spectrogram
    spec = get_spectrogram(data)
    spec_orig = spec.copy()
    spectrogram_size = spec.shape[0]

    # Load model
    # Get cpu or gpu device for training.
    mydevice = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {mydevice} device")
    device = torch.device(mydevice)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = NetDLD(num_tags, spectrogram_size)
    model.load_state_dict(torch.load(modelfile, map_location=device))
    model.eval()

    # Run model on audio
    spec = torch.from_numpy(spec)
    spec = spec.permute(1, 0)
    spec = spec.unsqueeze(0)
    y_pred = model(spec)
    y_pred_l = np.exp(y_pred[0].tolist())

    # Convert prediction into string
    # TODO: proper beam search
    m = torch.argmax(y_pred[0], 1)
    print(prediction_to_str(m))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model")
    # parser.add_argument("input")
    # args = parser.parse_args()
    # wavfile = args.input
    # modelfile = args.model

    # python decode_audio.py --model ./models/001750.pt ../CallCQ_pitch416_wpm17_noise83_amplitude36.wav
    # wavfile = "../data/cwWithWhiteNoiseSuper.wav"

    modelfile0 = "../models_lib/readme.R1.0_A.001800.pt"
    modelfile1 = "../models_lib/readme.R1.0_A.005000.pt"
    modelfile2 = "../models_lib/readme.R1.0_B.001880.pt"
    modelfile3 = "../models_lib/readme.R1.0_B.005090.pt"

    mList = []
    mList.append(modelfile0)
    mList.append(modelfile1)
    mList.append(modelfile2)
    mList.append(modelfile3)

    wavfile0 = "../sounds_lib/demo_with_mobile_reord_sound.wav"
    wavfile1 = "../sounds_lib/100MostCommonEnglishWords12.wav"
    wavfile2 = "../sounds_lib/100MostCommonEnglishWords24.wav"
    wavfile3 = "../sounds_lib/CallCQ_pitch545_wpm27_noise128_amplitude46.wav"

    sList = []
    sList.append(wavfile0)
    sList.append(wavfile1)
    sList.append(wavfile2)
    sList.append(wavfile3)

    # test case for multipule tests
    # choice one CW QSO, compared decode result by using different models
    print("Start to run test case ...")
    for s in sList:
        print("Using QSO : ", s)
        for m in mList:
            print("Using Model : ", m)
            decoder_file(m, s)

    print("Test case is completed.")
