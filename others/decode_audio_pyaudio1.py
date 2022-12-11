import pyaudio
import wave
from matplotlib import pyplot as plt


def paly_audio(data):
    color = [
        "red",
        "black",
        "yellow",
        "green",
        "pink",
        "gray",
        "lightgreen",
        "orange",
        "blue",
        "teal",
    ]
    # cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cls = [i for i in range(0, len(data))]
    data = [i for i in data]
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


def record_audio(wave_out_path, record_second):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    wf = wave.open(wave_out_path, 'wb')

    wf.setnchannels(CHANNELS)

    wf.setsampwidth(p.get_sample_size(FORMAT))

    wf.setframerate(RATE)

    print("设置参数完毕")
    print("开始录制声音")
    # int(RATE / CHUNK * record_second)录制时间内,需要读取多少次数据
    # CHUNK一次采样数据多少
    # RATE一秒采样多少
    for _ in range(0, int(RATE / CHUNK * record_second)):
        data = stream.read(CHUNK)

        # 这里直接迭代就会将data的16进制直接变成10进制
        # for one_data in data:
        #     print(one_data)
        paly_audio(data)
        wf.writeframes(data)

    print("录制完成")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


if __name__ == '__main__':
    record_audio("output.wav", record_second=4)
