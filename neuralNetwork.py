# -*- coding:utf-8 -*-
#训练代码
import numpy as np
import scipy.special
from struct import unpack
import matplotlib.pyplot as plt
import time
import  _pickle as cpickle

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

class neuralNetwork:
    #初始化
    def __init__(self,inputnodes,hiddennodes,outputnodes,learninggrate):
        self.inodes=inputnodes#输入层节点
        self.hnodes=hiddennodes#隐藏层节点
        self.onodes=outputnodes#输出层节点
        self.lr=learninggrate#学习率
        #链接权重矩阵
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))#input和hidden的权重矩阵
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))#hidden和output的权重矩阵
        pass
    #训练
    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets=np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.Sigmoid(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.Sigmoid(final_inputs)

        #误差
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors)
        #优化权重
        self.who+=self.lr*np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

        pass
    #查询
    def query(self,inputs_list):
        inputs=np.array(inputs_list,ndmin=2).T
        #把输入的list转换成numpy的array才能通过numpy模块进行矩阵运算。
        #numpy.array()的第一个参数为被转换的数组，ndmin参数为转换后的维数。
        #假设inputs_list的长度为len，通过np.array(inputs_list, ndmin=2)这
        # 条语句会被复制并转换为一个大小为(1,len)的矩阵。转置以后便是一个大小为(len,1)的矩阵

        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.Sigmoid(hidden_inputs)

        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.Sigmoid(final_inputs)
        return final_outputs

#    def activation_function(self,x):
    def Sigmoid(self,x):
        result = 1 / (1 + np.exp(-x))
        return result




if __name__ == '__main__':
    input_nodes=28*28
    hiddden_nodes=200
    output_nodes=10
    learning_rate=0.1

    n=neuralNetwork(input_nodes,hiddden_nodes,output_nodes,learning_rate)
    #打开文件，图片和标签
    img=read_image(
        "C:\\Users\Admin\\.PyCharm2017.1\\config\\"
        "scratches\\test\\t10k-images-idx3-ubyte\\"
        "t10k-images.idx3-ubyte")
    label = read_label(
        "C:\\Users\Admin\\.PyCharm2017.1\\config\\"
        "scratches\\test\\t10k-labels-idx1-ubyte\\"
        "t10k-labels.idx1-ubyte")
    # 训练集图像：t10k-images.idx3-ubyte
    # 训练集标签：t10k-labels.idx1-ubyte
    # 测试集图像：train-images.idx3-ubyte
    # 测试集标签：train-labels.idx1-ubyte
    inputs = img / 255.0 * 0.99 + 0.01
    l=len(label)
    #开始训练
    epochs=5
    start = time.time()
    for e in range(epochs):
        for record in range(0,l):
            targets=np.zeros(output_nodes)+0.01
            targets[label[record]]=0.99
            n.train(inputs[record],targets)
            pass
        pass
    end = time.time()

    with open('testwih.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        cpickle.dump([n.wih], f,0)
    f.close()
    with open('testwho.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
         cpickle.dump([n.who], f,0)
    f.close()


    print("训练时长:%.2f秒" % (end - start))


