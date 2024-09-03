# -*- coding: utf-8 -*-
# @Author  : Forerunner
# @Time    : 2024-09-03 9:22
# @File    : 数字识别.py
# @Software: PyCharm
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
from numModelNet import Net
"""
卷积运算 使用mnist数据集，和10-4，11类似的，只是这里：1.输出训练轮的acc 2.模型上使用torch.nn.Sequential
"""
# Super parameter ------------------------------------------------------------------------------------
batch_size = 64
learning_rate = 0.01
momentum = 0.5  # 冲量
EPOCH = 15

# Prepare dataset ------------------------------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

# 使用元组可以方便地处理不同通道的均值和标准差。例如
# ，对于 RGB 图像，可以使用 Normalize((mean_R, mean_G, mean_B), (std_R, std_G, std_B))
# 来分别指定每个通道的均值和标准差。

# softmax归一化指数函数(https://blog.csdn.net/lz_peter/article/details/84574716),其中0.1307是mean均值和0.3081是std标准差

train_dataset = datasets.MNIST(root='../data/mnist', train=True, transform=transform,
                               download=True)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='../data/mnist', train=False, transform=transform,
                              download=True)  # train=True训练集，=False测试集
print(f'训练集长度为', len(train_dataset))  # 60000
# batchsize 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(len(train_loader))  # 938

# for x,y in train_loader:
#     len+=1
#     print(x.shape) # [64, 1, 28, 28] 图像数据
#     print(y.shape) # [64]  图像对应的数字
# print("len is ",len)  # 938
# print(len*batch_size) # 60032

fig = plt.figure()
for i in range(1, 37):
    plt.subplot(6, 6, i)
    plt.tight_layout()
    plt.imshow(train_dataset.data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

device = torch.device("cuda")

# 训练集乱序，测试集有序
# Design model using class ------------------------------------------------------------------------------

model = Net()
model.to(device)

print(model)

# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量

# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    count = 0
    for data in train_loader:
        count += 1
        inputs, target = data
        inputs=inputs.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        # 模型预测出来的数据为 [0.1,0.1,0.1,0.005,0.005,0.005,0.8,0.1,0.1,0.2,0.02] 代表数字0-9的概率 我们需要找到最大概率的下标 然后和label值进行对比 可以找出模型的精确程度
        _, predicted = torch.max(outputs.data, dim=1)  # torch.max(outputs.data, dim=1)找出的是行最大值以及对应的下标
        running_total += inputs.shape[0]  # 行高 64
        running_correct += (predicted == target).sum().item()

        if count % 300 == 0:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print(f'训练轮次:{epoch},该轮训练次数:{count + 1} loss:{running_loss} , acc: {100 * running_correct / running_total}%'
                 )
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch + 1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(1,EPOCH+1):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)
    torch.save(model, "NumClassifierModel.pth")
    print("模型已经保存")
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
