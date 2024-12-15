# 1 加载必要的库

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 2 定义超参数
BATCH_SIZE = 1024 # 每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 是否用GPU
EPOCHS = 10 # 训练数据的轮次

# 3 构建pipeline，对图形做处理
pipeline = transforms.Compose([
    transforms.ToTensor(), # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 正则化：降低模型复杂度
])

# 4 下载、加载数据集
"""
MNIST数据集会被下载到当前工作目录下的 "data" 文件夹中。
如果你希望更改存储路径，只需将 "data" 替换为你想要的其他路径即可
MNIST数据集是从PyTorch的官方数据集中下载的。
具体来说，datasets.MNIST会自动从 Yann LeCun's website 下载MNIST数据集
如果你已经下载过MNIST数据集并且数据文件存在于指定的存储路径中，datasets.MNIST 将不会再次下载这些文件
"""
from torch.utils.data import DataLoader
# 下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
# 解释：
# - "data": 指定数据集存储的目录。
# - train=True: 表示下载的是训练数据集。
# - download=True: 表示如果数据集不存在，则进行下载。
# - transform=pipeline: 对下载的数据应用预处理步骤（如转换为Tensor和标准化）。
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)
# 解释：
# - "data": 指定数据集存储的目录。
# - train=False: 表示下载的是测试数据集。
# - download=True: 表示如果数据集不存在，则进行下载。
# - transform=pipeline: 对下载的数据应用预处理步骤（如转换为Tensor和标准化）。

# 加载数据集
# - shuffle=True: 在每个epoch开始时打乱数据顺序
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
"""
#插入代码，显示图片
with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    file = f.read()
image1 = [int(str(item).encode('ascii'), 16) for item in file[16 : 16+784]]
print(image1)

import cv2
import numpy as np
image1_np = np.array(image1, dtype=np.uint8).reshape(28, 28, 1)
print(image1_np.shape)
#保存
cv2.imwrite("digit.png", image1_np)
"""

# 5 构建网络模型
"""
nn.Module 是 PyTorch 中所有神经网络模块的基类，提供了参数管理、前向传播、状态切换等功能。
"""
class Digit(nn.Module):
    def __init__(self):
        super().__init__()#super().__init__():：调用父类 nn.Module 的构造函数，确保正确初始化基类的部分。
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1: 灰度图片的通道，10：输出通道，5：kernel
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道，20: 输出通道，3: kernel
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 20*10*10: 输入通道， 500: 输出通道
        self.fc2 = nn.Linear(500, 10)  # 500: 输入通道，10:输出通道

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)  # 输入: batch*1*28*28, 输出: batch*10*24*24 （28-5+1）
        x = F.relu(x)  # 保持shape不变，输出：batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 输入：batch*10*24*24, 输出: batch*10*12*12
        x = self.conv2(x)  # 输入: batch*10*12*12, 输出: batch*20*10*10
        x = F.relu(x)  # 保持shape不变，输出：batch*20*10*10
        x = x.view(input_size, -1)  # 拉平，-1 自动计算维度， 20*10*10=2000
        x = self.fc1(x)  # 输入: batch*2000, 输出: batch*500
        x = F.relu(x)  # 保持shape不变，输出: batch*500
        x = self.fc2(x)  # 输入: batch*500, 输出: 10
        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值

        return output

# 6 定义优化器
model = Digit().to(DEVICE)#模型部署到设备上
optimizer = optim.Adam(model.parameters())#优化器更新模型参数

# 7 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        """
        enumerate(train_loader):
        enumerate 是一个 Python 内置函数，用于将一个可遍历的数据对象（如列表、元组或迭代器）组合为一个索引序列，同时列出数据和数据下标。
        train_loader 通常是一个数据加载器（DataLoader），它负责从训练数据集中批量读取数据。
        每个批次包含一组输入数据 (data) 和对应的标签 (target)。
        batch_index 是当前批次的索引，表示这是第几个批次。
        (data, target) 是当前批次中的数据和标签：
        data: 输入数据张量（Tensor），通常是图像、文本或其他类型的数据。
        target: 标签张量（Tensor），通常是与输入数据相对应的真实类别或目标值。
        """
        # 部署到device上
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)#这只是一个计算损失的函数，适用于计算多分类的任务
        #找到概率值最大的下标
        pred = output.max(1, keepdim=True)#其中的1表示维度
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))#这里需要用loss.item才能取到值

# 8 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    # 不会进行计算梯度，也不会进行反向传播
    with torch.no_grad():
        for data, target in test_loader:
            # 部署到device上
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[1]
            #pred = torch.max(output, dim=1)
            # pred = output.argmax(dim=1)
            # 累计正确的值
            correct += pred.eq(target.view_as(pred)).sum().item()
        # 计算平均loss
        test_loss /= len(test_loader.dataset)
        print("Test -- Average Loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, correct/len(test_loader.dataset)*100.0))

# 9 调用方法(7和8)
for epoch in range(1, EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)