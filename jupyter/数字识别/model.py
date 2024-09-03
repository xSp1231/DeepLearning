# -*- coding: utf-8 -*-
# @Author  : Forerunner
# @Time    : 2024-09-03 13:38
# @File    : model.py
# @Software: PyCharm
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, padding=2),
                nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


if __name__ == '__main__':
    print(Net)
    print("模型启动成功")
