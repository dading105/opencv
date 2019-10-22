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




通过高斯滤波，边缘检测和findContours，识别瓷瓶污秽，生锈程度，受潮情况，缓冲器渗漏等情况

并使用隐马尔科夫链，智能预测电力设备疲劳，老化的情况

技术交流群：334312796
验证密码：深圳友先达
