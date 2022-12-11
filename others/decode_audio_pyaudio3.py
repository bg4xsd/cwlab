import pyaudio
import wave
from tqdm import tqdm
from matplotlib import pyplot as plt


def paly_audio(data):
    color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen", "orange", "blue", "teal"]
    data = [i for i in data]

    cls = [i for i in range(0, len(data))]
    plt.ion()
    plt.clf()
    plt.plot(cls, data, "-", color=color[0])
    # 这里所有图片的路径已经分布位置弄出来统计按照分类列表分别存起来,最后计算每个类别的范围L,将L分为3个部分,中间部分假设为真是分布
    # 边缘部分在数据多的情况下去掉,如果数据少可以将边缘部再分几部分,去掉边缘部分即可
    # print(x,y)
    # plt.legend(label, loc="upper right")  # 如果写在plot上面，则标签内容不能显示完整
    # plt.title("epoch={}".format(str(epoch + 1)))
    # plt.savefig('{}/{}.jpg'.format(save_path, epoch + 1))
    plt.draw()
    plt.pause(0.01)


def play_audio(wave_path):
    CHUNK = 1024
    wf = wave.open(wave_path, 'rb')
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data
    data = wf.readframes(CHUNK)
    # play stream (3)
    datas = []
    while len(data) > 0:
        data = wf.readframes(CHUNK)
        datas.append(data)
    for d in tqdm(datas):
        paly_audio(d)
        stream.write(d)
    # stop stream (4)
    stream.stop_stream()
    stream.close()
    # close PyAudio (5)
    p.terminate()


if __name__ == '__main__':
    play_audio("output.wav")

