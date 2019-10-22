# opencv
泛在电力物联网

配电房入侵人脸抓拍
python3语言
调用opencv face_recognition tensorflow cvlib库

#导入图片



# Import the image
img = cv2.imread('burano.jpg')

#灰度



# Convert the image into gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#HSV HLS



# Transform the image into HSV and HLS models
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGi2HSV)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img, 50, 150)

#设备特征识别

import cv2

import matplotlib.pyplot as plt

import cvlib as cv

from cvlib.object_detection import draw_bbox

im = cv2.imread('cars_4.jpeg')

bbox, label, conf = cv.detect_common_objects(im)

output_image = draw_bbox(im, bbox, label, conf)

plt.imshow(output_image)

plt.show()

print('Number of cars in the image is '+ str(label.count('distribution box')))


states = ('Rainy', 'Sunny')
observations = ('walk', 'shop', 'clean')
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }
emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}
上图是一个简单的HMM。我们的目标是要从表状态链（生活）中找到最可能的隐状态链（天气）。 
传统的维特比算法，等价于将HMM展开为概率图，最好的隐状态链就是概率图中的最优路径对应的隐状态链，所以维特比算法思路等价于dijistra最优路径算法。这里要说明几个要点： 
- 概率低不代表不发生 
- 节点的某个状态概率低，依然可以选择这条路径节点，因为我们求解的是全局最优路径 
- 在求解最优隐状态链时，我们实际上给隐状态链加入了一个门阀矩阵Pk，比如选取状态二就是点乘[0,0,1]矩阵
HMM隐马尔可夫模型，即通过统计的方法可以去观察和认知一个事件序列上邻近事件发生的概率转换问题。
如何训练HMM模型:输入Xi序列和Oi序列，全部通过统计学模型完成，得到的模型结果就是一个转移矩阵。一个输出概率矩阵和一个隐含状态转移矩阵。这样可以对下一个输出状态进行基于概率的预测。


通过高斯滤波，边缘检测和findContours，识别瓷瓶污秽，生锈程度，受潮情况，缓冲器渗漏等情况

并使用隐马尔科夫链，智能预测电力设备疲劳，老化的情况

技术交流群：334312796
验证密码：深圳友先达
