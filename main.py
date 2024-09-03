import torch
import numpy as np
import matplotlib.pyplot as plt

# 创建一个随机 RGB 图像的张量 (3, 28, 28)
image_tensor = torch.rand(1, 28, 28)  # 形状为 (C, H, W)

# 转换为 NumPy 数组并调整形状为 (H, W, C)
image_np = image_tensor.permute(1, 2, 0).numpy()  # 变为 (28, 28, 3)

# 使用 imshow 绘制图像
plt.imshow(image_np)
plt.axis('off')  # 隐藏坐标轴
plt.title("Random RGB Image")
plt.show()