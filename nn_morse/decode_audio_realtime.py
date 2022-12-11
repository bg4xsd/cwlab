#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  decode_audio_realtime.py
@Time    :  :2022/12/11
@Author  :   Dr. Cat Lu / BFcat
@Version :   1.0
@Contact :   bfcat@live.cn
@Site    :   https://bg4xsd.github.io
@License :   (C)MIT License
@Desc    :   This is a part of project XXX, more details can be found on the site.
'''
import os
import sys

os.chdir(sys.path[0])
print("Current work directory -> %s" % os.getcwd())

import numpy as np
import random
import pyaudio
import decode_audio as da
from matplotlib import pyplot as plt
from morse import ALPHABET, generate_sample

modelfile = "./models/001750.pt"
wavfile = "../sounds/testaudio.wav"

'''
char 0  use 4664 byte
char 1  use 4412 byte
char E  use 2049 byte

Five 0  use 17255 byte, 
callsignal bg4xsd/dx  18639 yte
callsignal bg4xsd/qrp 20622 byte
'''


def Monitor(record_second):
    CHUNK = 30720
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 2000
    SAMPLE_FREQ = 2000
    WAVE_OUTPUT_FILENAME = "../temp/cache.wav"
    WAVE_OUTPUT_FILENAME = "../temp/cache.wav"

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    # TODO 这里可以生成一个数字倒计时，然后开始录音，顺便测试 decode是否正常工作
    length = random.randrange(10, 20)
    pitch = random.randrange(100, 950)
    wpm = random.randrange(10, 30)
    noise_power = random.randrange(0, 200)
    amplitude = random.randrange(10, 150)

    s = "CQ CQ CQ DE BG4XSD BG4XSD PSE K E E"
    s = s.upper()  # must be upcase
    samples, spec, y = generate_sample(length, pitch, wpm, noise_power, amplitude, s)
    samples = samples.astype(np.float32)

    da.decoder(modelfile, samples)

    print("If you see the CQ message, the decoder should work well.\n")

    print("Begine to read from Mic ...")
    print("Press D to stop ...")

    while True:
        data = stream.read(CHUNK)
        # da.decoder(modelfile, frames)
        audio_data = np.frombuffer(data, dtype=np.short)
        da.decoder(modelfile, audio_data)

    print("Stop monitoring and release resource ...")
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    Monitor(record_second=20)

    # import pandas as pd
    # a = list(audio_data)
    # name_attribute = ['a']
    # writerCSV=pd.DataFrame(columns=name_attribute,data=a)
    # writerCSV.to_csv('./raw_mic.csv',encoding='utf-8')
