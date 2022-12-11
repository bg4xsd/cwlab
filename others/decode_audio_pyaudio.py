#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  decode_real_audio.py
@Time    :  :2022/12/10
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

import pyaudio
import wave
import numpy as np


def Monitor():
    CHUNK = 1024
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 2000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "../temp/cache.wav"
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    print("Begine to read from Mic ...")
    frames = []
    while True:
        print('begin ')
        for i in range(0, 100):
            data = stream.read(CHUNK)
            frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.short)
        large_sample_count = np.sum(audio_data > 800)
        temp = np.max(audio_data)
        if temp > 100:
            print("Signal detected ...")
            print('Current threshold : ', temp)
            break
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


if __name__ == '__main__':
    Monitor()



# import pandas as pd

# name_attribute = ['a']
# writerCSV=pd.DataFrame(columns=name_attribute,data=a)

# writerCSV.to_csv('./no_fre.csv',encoding='utf-8')