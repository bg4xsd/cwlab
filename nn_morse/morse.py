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

SAMPLE_FREQ = 2000  # 2 Khz，6Khz and 8khz can be deall with other software

# 59 Chars dict
MORSE_CODE_DICT = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    '0': '-----',
    '.': '.-.-.-',
    ',': '--..--',
    '?': '..--..',
    "'": '.----.',
    '!': '-.-.--',
    '/': '-..-.',
    '(': '-.--.',  # KN
    ')': '-.--.-',
    '&': '.-...',  # AS
    ':': '---...',
    ';': '-.-.-.',
    '=': '-...-',
    '+': '.-.-.',  # AR
    '-': '-....-',
    '_': '..--.-',
    '"': '.-..-.',
    '$': '...-..-',
    '@': '.--.-.',
    '$': '...-.-',
    '#': '-...-.-',  # BK replace
    '%': '-.-..-..',  # CL replace
    '^': '-...-',  # BT replace
    '*': '...-.-',  # SK replace
}
ALPHABET = " " + "".join(MORSE_CODE_DICT.keys())


def get_spectrogram(samples):
    # 250 units/min，也就是 60 sec/250unit, 240ms/unit @ 5WPM，
    # 120ms/unit @ 10WPM，60ms/unit @20WPM，30ms/unit @40WPM。
    # 24ms/unit @50WPM, 这个窗口勉强覆盖到 50 WPM， 应该是够用了。
    # Resolution = windows size  / sample rate
    # If windows size is  256, 256/6000=0.043, is  43ms
    # So, the windows size = resolution x sample rate
    # Aslo, we can enlarge or reduce it.
    # Ref : https://blog.csdn.net/qq_29884019/article/details/106177650
    window_length = int(0.02 * SAMPLE_FREQ)  # 20 ms windows

    # Ref : https://blog.csdn.net/zhuoqingjoking97298/article/details/122634775
    # smaples is the collected dataset/time series
    # nperseg 每个段的长度。默认为无，但是如果window是str或tuple，
    #         则设置为256，如果window是数组，则设置为窗口的长度。
    # Return value list:
    #   f：ndarray   采样频率数组
    #   t：ndarray   细分时间数组
    #   Sxx：ndarray   x的频谱图,默认情况下，Sxx的最后一个轴对应于段时间。
    _, _, s = signal.spectrogram(samples, nperseg=window_length, noverlap=0)
    return s


def generate_sample(
    text_len=10, pitch=500, wpm=20, noise_power=1, amplitude=100, s=None
):
    assert pitch < SAMPLE_FREQ / 2  # Nyquist

    # Reference word is PARIS, 50 dots long
    dot = (60 / wpm) / 50 * SAMPLE_FREQ

    # Add some noise on the length of dash and dot
    #
    # np.random.randn(np.random.randint(4, 7))
    # More difficult for NN
    #
    # 设置的过宽，会导致dot，dash无法识别，现在缩小的波动范围
    sl = 0.5  # scale low limit，原来是0.7
    su = 2.0  # scale up limit，原来是2

    def get_dot():
        scale = np.clip(np.random.normal(1, 0.2), sl, su)
        return int(dot * scale)

    # The length of a dash is three times the length of a dot.
    def get_dash():
        scale = np.clip(np.random.normal(1, 0.2), sl, su)
        return int(3 * dot * scale)

    # Create random string that doesn't start or end with a space
    # In this version, ' ' is added into the dict, so the apace will appear.
    if s is None:
        if text_len == 1:
            s2 = ''.join(random.choices(ALPHABET[1:], k=2))
            s = s2[1]
        else:
            s1 = ''.join(random.choices(ALPHABET, k=text_len - 2))
            s2 = ''.join(random.choices(ALPHABET[1:], k=2))
            s = s2[0] + s1 + s2[1]
    else:
        s = s

    out = []
    # The original author choose to use 5 * get_dot()
    # out.append(np.zeros(5 * get_dot()))
    # For clear CW environment, 0 for no signal, 1 for signal
    # then, the noise will be added.
    # Another problem, we can not make sure the distance to last words
    # So, I use a random int to indicate the distance, not the fixed value.
    out.append(np.zeros(random.randint(1, 7) * get_dot()))

    # The space between two signs of the same character is equal to the length of one dot.
    # The space between two characters of the same word is three times the length of a dot.
    # The space between two words is seven times the length of a dot (or more).
    s = s.upper()
    for c in s:
        if c == ' ':
            # total 7 unit for white space
            out.append(np.zeros(get_dot() + 2 * get_dot() + 3 * get_dot() + get_dot()))
        else:
            for m in MORSE_CODE_DICT[c]:
                if m == '.':
                    out.append(np.ones(get_dot()))
                    out.append(np.zeros(get_dot()))
                elif m == '-':
                    out.append(np.ones(get_dash()))
                    out.append(np.zeros(get_dot()))

            out.append(np.zeros(get_dot() + get_dot()))

    # The original author choose to use 5 * get_dot(), at the end of a word
    # out.append(np.zeros(5 * get_dot()))
    # There are already 3 unit at the end of last char, so add 4 units will
    # indicate the end of a word. Here I use a random int 3~5
    out.append(np.zeros(random.randint(3, 5) * get_dot()))

    # Convert to one line array
    out = np.hstack(out)

    # Modulatation
    t = np.arange(len(out)) / SAMPLE_FREQ
    sine = np.sin(2 * np.pi * t * pitch)
    out = sine * out

    # If you want to see the modulated data,run below code in Jupyter
    # import matplotlib.pyplot as plt
    # x=range(len(sine))
    # plt.plot(x,sine)

    # Add noise
    noise_power = 1e-6 * noise_power * SAMPLE_FREQ / 2
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(out))
    out = 0.5 * out + noise

    out *= amplitude / 100
    out = np.clip(out, -1, 1)

    out = out.astype(np.float32)

    spec = get_spectrogram(out)

    return out, spec, s


if __name__ == "__main__":
    from scipy.io.wavfile import write
    import matplotlib.pyplot as plt

    debug = False

    if not debug:
        length = random.randrange(10, 20)
        pitch = random.randrange(100, 950)
        wpm = random.randrange(10, 30)
        noise_power = random.randrange(0, 200)
        amplitude = random.randrange(10, 150)

        s = "CQ CQ CQ DE BG4XSD BG4XSD PSE K E E"
    else:
        length = 2
        pitch = 650
        wpm = 20
        noise_power = 0
        amplitude = 50
        # s = 'paris'
        s = None
    samples, spec, y = generate_sample(length, pitch, wpm, noise_power, amplitude, s)
    print("Spec shape is : ", spec.shape)
    print(f"pitch: {pitch} wpm: {wpm} noise: {noise_power} amplitude: {amplitude}")
    print(
        "sentence is :",
        y,
        "length is :",
        length,
        " smaples size is :",
        len(samples),
        'wpm is ',
        wpm,
    )
    samples = samples.astype(np.float32)
    fname = (
        "CallCQ_pitch"
        + str(pitch)
        + "_wpm"
        + str(wpm)
        + "_noise"
        + str(noise_power)
        + "_amplitude"
        + str(amplitude)
        + ".wav"
    )

    write("../temp/testaudio.wav", SAMPLE_FREQ, samples)

    plt.figure()
    plt.pcolormesh(spec)
    plt.show()
