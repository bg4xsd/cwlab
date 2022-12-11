import soundcard as sc

# get a list of all speakers:
speakers = sc.all_speakers()
# get the current default speaker on your system:
default_speaker = sc.default_speaker()
# get a list of all microphones:
mics = sc.all_microphones()
# get the current default microphone on your system:
default_mic = sc.default_microphone()


import numpy
import matplotlib.pyplot as plt

default_mic = sc.default_microphone()
# record and play back one second of audio:
# data = default_mic.record(samplerate=48000, numframes=48000)

with default_mic.recorder(samplerate=48000) as mic:
    mic.record(numframes=1000)
    data = mic.record(numframes=48000)
    plt.plot(data)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
default_speaker.play(data / numpy.max(data), samplerate=48000)


# alternatively, get a `Recorder` and `Player` object
# and play or record continuously:
with default_mic.recorder(samplerate=48000) as mic, default_speaker.player(
    samplerate=48000
) as sp:
    for _ in range(100):
        data = mic.record(numframes=10240)
        sp.play(data)




