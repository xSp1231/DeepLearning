# -*- coding: utf-8 -*-
# @Author  : Forerunner
# @Time    : 2024-09-03 11:26
# @File    : 数字识别测试.py
# @Software: PyCharm


import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import datasets
from torchvision import transforms
from numModelNet import Net
from PIL import Image
img_path = "testnum/11.png"  # jpg图片3个通道 png图片4个通道 多了一个透明度通道
image = Image.open(img_path)
image = image.convert("L")
image = Image.eval(image, lambda x: 255 - x)
size = 28
transform = transforms.Compose([transforms.Resize((size, size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]
                               )
print(f"转换之前", image.size)
image = transform(image)

# test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform,
#                               download=True)  # train=True训练集，=False测试集
#
model = torch.load("NumClassifierModel.pth", map_location=torch.device("cpu"))  # 引入模型 模型在cpu上进行测试
#
count = 0
#
model.eval()
with torch.no_grad():
        print(image.shape)
        # image的维度是 1*28*28
        # 使用 squeeze 去掉第一个维度，使形状变为 [28, 28]
        image = image.squeeze(0)  # 现在形状为 [28, 28]
        plt.imshow(image, cmap='gray', interpolation='none')  # 使用灰度色图
        plt.title(f"Label")  # 显示标签
        plt.axis('off')  # 隐藏坐标轴
        plt.show()  # 显示图像
        image = image.reshape(1, 1, 28, 28)
        output = model(image)
        max_index = torch.argmax(output, dim=1)  # 按照列的方向来寻找
        print("预测值 is ", max_index.item())









#
# model.eval()
# with torch.no_grad():
#     for image, label in test_dataset:
#         count += 1
#         print(image.shape)
#         # image的维度是 1*28*28
#         # 使用 squeeze 去掉第一个维度，使形状变为 [28, 28]
#         image = image.squeeze(0)  # 现在形状为 [28, 28]
#         plt.imshow(image, cmap='gray', interpolation='none')  # 使用灰度色图
#         plt.title(f"Label: {label}")  # 显示标签
#         plt.axis('off')  # 隐藏坐标轴
#         plt.show()  # 显示图像
#         image = image.reshape(1, 1, 28, 28)
#         output = model(image)
#         max_index = torch.argmax(output, dim=1)  # 按照列的方向来寻找
#         print("预测值 is ", max_index.item(), "label is ", label)
#         if count == 10:
#             break
