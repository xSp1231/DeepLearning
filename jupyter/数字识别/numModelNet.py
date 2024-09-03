# -*- coding: utf-8 -*-
# @Author  : Forerunner
# @Time    : 2024-09-03 11:36
# @File    : numModelNet.py
# @Software: PyCharm
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # 卷积的步幅默认为1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 池化的步幅默认为池化层的宽高大小
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.model(x)  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


if __name__ == '__main__':
    print(Net)
    print("模型启动成功")
