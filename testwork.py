# -*- coding:utf-8 -*-
#测试代码
import numpy as np
import  _pickle as cpickle
from struct import unpack
import matplotlib.pyplot as plt
import scipy.special
#读取图片文件
def read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

#读取标签文件
def read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab

def query(wih,who, inputs_list):
    inputs = np.array(inputs_list, ndmin=2).T
    hidden_inputs = np.dot(wih, inputs)
    hidden_outputs = Sigmoid(hidden_inputs)
    final_inputs = np.dot(who, hidden_outputs)
    final_outputs = Sigmoid(final_inputs)
    return final_outputs

def Sigmoid(x):
     result = 1 / (1 + np.exp(-x))
     return result

if __name__ == '__main__':
    img_test = read_image(
        "C:\\Users\Admin\\.PyCharm2017.1\\config\\"
        "scratches\\test\\train-images-idx3-ubyte\\"
        "train-images.idx3-ubyte")
    label_test = read_label(
        "C:\\Users\Admin\\.PyCharm2017.1\\config\\"
        "scratches\\test\\train-labels-idx1-ubyte\\"
        "train-labels.idx1-ubyte")
    with open('testwho.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        who = cpickle.load(f)
    f.close()
    with open('testwih.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        wih = cpickle.load(f)
    f.close()
    l_test = len(label_test)
    inputs_test = img_test / 255.0 * 0.99 + 0.01
    score = 0
    print("开始测试训练集中的数据，请耐心等待。")
    for record in range(0, 1):
        g = query(wih,who,inputs_test[record])
        label_get = np.argmax(g)
        if (label_get == label_test[record]):
            score += 1

    score /= l_test
    print("正确率=", score)

#单张测试，w是数据集中第w张照片，修改w值可以看到数据集中的图片和识别出来的数字
    w = input("输入想看的照片w")
    w=int(w)
    gg = query(wih,who,inputs_test[w])
    print("实际数字为",label_test[w])
    num = np.argmax(gg)
    print("测试得到",num)
    plt.imshow(img_test[w].reshape((28, 28)), 'Greys', interpolation='None')
    plt.show()