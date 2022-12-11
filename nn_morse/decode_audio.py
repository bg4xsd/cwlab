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


def decoder(modelfile, data):

    if len(modelfile) < 1:
        modelfile = "./models/001750.pt"

    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # Create spectrogram
    spec = get_spectrogram(data)
    spec_orig = spec.copy()
    spectrogram_size = spec.shape[0]

    # Load model
    device = torch.device("cpu")
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


def decoder_file(modelfile, wavfile):
    if len(wavfile) < 1:
        wavfile = "../sounds/testaudio.wav"
    if len(modelfile) < 1:
        modelfile = "./models/001750.pt"

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
    device = torch.device("cpu")
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

    modelfile = "./models/001750.pt"
    # wavfile = "../sounds/testaudio.wav"
    wavfile = "../temp/cache.wav"

    decoder_file(modelfile, wavfile)

    # python decode_audio.py --model ./models/001750.pt ../CallCQ_pitch416_wpm17_noise83_amplitude36.wav
    # wavfile = "../data/cwWithWhiteNoiseSuper.wav"
