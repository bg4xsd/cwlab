#!/usr/bin/env python3

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

from main import Net, num_tags, prediction_to_str
from morse import ALPHABET, SAMPLE_FREQ, get_spectrogram

# For example
# python decode_audio.py --model ./models/001750.pt ../CallCQ_pitch416_wpm17_noise83_amplitude36.wav


def decoder_realtime(modelfile, data):
    print("TODO")


def decoder_file(modelfile, wavfile):
    if len(wavfile) < 1:
        wavfile = "../sounds_lib/CQ_DE_BG4XSD.wav"
    if len(modelfile) < 1:
        modelfile = "../models_lib/original_NNmodel_001750.pt"

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
    device = torch.device("cuda")
    model = Net(num_tags, spectrogram_size)
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

    # # Only show letters with > 5% prob somewhere in the sequence
    # labels = np.asarray(["<blank>", "<space>"] + list(ALPHABET[1:]))
    # sum_prob = np.max(y_pred_l, axis=0)
    # show_letters = sum_prob > 0.05

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.pcolormesh(spec_orig)
    # plt.subplot(2, 1, 2)
    # plt.plot(y_pred_l[:, show_letters])
    # plt.legend(labels[show_letters])
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model")
    # parser.add_argument("input")
    # args = parser.parse_args()
    # wavfile = args.input
    # modelfile = args.model

    # python decode_audio.py --model ./models/001750.pt ../CallCQ_pitch416_wpm17_noise83_amplitude36.wav
    # wavfile = "../data/cwWithWhiteNoiseSuper.wav"

    modelfile0 = "../models_lib/demo_NNmodel_001750.pt"
    modelfile1 = "../models_lib/models_Len1-20_7.5k_varLRx0.5/005000.pt"
    modelfile2 = "../models_lib/models_Len1-20_6.0k_varLRx0.9/006000.pt"
    modelfile3 = "../models_lib/models_Len1_20_batch64_14k_5stage/007000.pt"
    modelfile4 = "../models_lib/models_Len1_20_batch128_14k_5stage/008000.pt"

    mList = []
    mList.append(modelfile0)
    mList.append(modelfile1)
    mList.append(modelfile2)
    mList.append(modelfile3)
    mList.append(modelfile4)

    wavfile0 = "../sounds_lib/CallCQ_pitch711_wpm15_noise142_amplitude106.wav"
    wavfile1 = "../sounds_lib/CallCQ_pitch479_wpm20_noise34_amplitude115.wav"
    wavfile2 = "../sounds_lib/CallCQ_pitch369_wpm25_noise31_amplitude88.wav"
    wavfile3 = "../sounds_lib/CallCQ_pitch573_wpm29_noise47_amplitude59.wav"
    wavfile4 = "../sounds_lib/demo_with_mobile_reord_sound.wav"
    wavfile5 = "../sounds_lib/realQSO_001.wav"
    wavfile6 = "../sounds_lib/realQSO_002.wav"
    wavfile7 = "../sounds_lib/realQSO_003.wav"

    sList = []
    sList.append(wavfile0)
    sList.append(wavfile1)
    sList.append(wavfile2)
    # sList.append(wavfile3)
    # sList.append(wavfile4)
    sList.append(wavfile5)
    sList.append(wavfile6)
    sList.append(wavfile7)

    # single compare test
    # decoder_file(modelfile0, wavfile0)
    # decoder_file(modelfile1, wavfile1)

    # test case for multipule tests
    # choice one CW QSO, compared decode result by using different models
    print("Start to run test case ...")
    for s in sList:
        print("Using QSO : ", s)
        for m in mList:
            print("Using Model : ", m)
            decoder_file(m, s)

    print("Test case is completed.")
