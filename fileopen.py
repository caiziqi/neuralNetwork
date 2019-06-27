from tkinter import *
import matplotlib.pyplot as plt
import scipy.special
import  _pickle as cpickle
from struct import unpack
import numpy as np
import cv2




drawing = False # 鼠标左键按下时，该值为True，标记正在绘画
mode = True # True 画矩形，False 画圆
ix, iy = -1, -1 # 鼠标左键按下时的坐标
flag=0



def query(wih,who, inputs_list):
    inputs = np.array(inputs_list, ndmin=2).T
    # 把输入的list转换成numpy的array才能通过numpy模块进行矩阵运算。
    # numpy.array()的第一个参数为被转换的数组，ndmin参数为转换后的维数。
    # 假设inputs_list的长度为len，通过np.array(inputs_list, ndmin=2)这
    # 条语句会被复制并转换为一个大小为(1,len)的矩阵。转置以后便是一个大小为(len,1)的矩阵

    hidden_inputs = np.dot(wih, inputs)
    hidden_outputs = Sigmoid(hidden_inputs)

    final_inputs = np.dot(who, hidden_outputs)
    final_outputs = Sigmoid(final_inputs)
    return final_outputs

def Sigmoid(x):
     result = 1 / (1 + np.exp(-x))
     return result



def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        # 鼠标移动事件
        if drawing == True:
            if mode == False:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 15, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP :
        # 鼠标左键松开事件
        drawing = False



with open('testwho.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    who = cpickle.load(f)
f.close()
with open('testwih.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    wih = cpickle.load(f)
f.close()


img = np.ones([280, 280, 1], np.uint8)
img = img * 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle) # 设置鼠标事件的回调函数
num=0

while(num<1200):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    num+=1
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
#在键盘上按m切换模式
print("开始写数字")
#img=img.reshape((1,78400))
#img=np.concatenate(img, axis=0)

img_test=np.zeros((28,28))
for i in range(0,28):
    for j in range(0,28):
        img_test[i][j]=img[i*10:i*10+10,j*10:j*10+10].sum()/100
img_test=img_test.reshape((1,784))
img_test=255.0-img_test
inputs_test = img_test / 255.0 * 0.99 + 0.01
gg = query(wih,who,inputs_test)
num = np.argmax(gg)
print("识别出来数字为：",num)
plt.imshow(inputs_test.reshape((28, 28)), 'Greys', interpolation='None')
plt.show()

