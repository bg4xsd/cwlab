## CW Lab

![](./img/hamcodes-640x336.jpg)

&nbsp;&nbsp;CW stands for Continuous Wave. In CW communications, a continuous carrier is transmitted as long as the key contacts are closed. By making and breaking the contacts into specific patterns, characters and entire messages may be sent.

&nbsp;&nbsp;Here is some work about Ham CW, Morse code, by using Nerual Network, Machine Learning method help us learn, recognize the code.

&nbsp;&nbsp;&nbsp;**&nbsp;nn_morse** is implemented by pd0wm almost 3 years ago, more details can be found at <https://github.com/pd0wm>. It works well now, with pyton3.9, pytorch 1.13 + cuda 11.7. It's very useful to broaden vision on the data generating and neural network traning.

&nbsp;&nbsp;**morse-datase**t is another open source project by souryadey, almost 5 year ago, which is very useful, too. He suggests a method to distinguish 0 and 1 in morse code, by using stochastical method. More detailed introduction can be found at <https://github.com/souryadey/morse-dataset>.

&nbsp;&nbsp;Also, I'd like to add some features on code generating, then use it with matlab, the new code will be collected in  CWlab. Here, ml, nn and dsp are to be used, different technical roadmap. In general,  I will try to implement realtime audio input, and with realtime text message as output.

&nbsp;&nbsp;Some notes:

1. The directory **models_lib** is used to store some pre-trained models, the name would be useful to indicate where it come from.

2. The directory of **sounds_lib** is used to store some example. Real QSO seems somewaht difficult to the Models.

Just for fun,  keep it simple.

By BFcat @  2022.12.10
