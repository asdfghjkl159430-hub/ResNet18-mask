
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 定义图像预处理（不要提前 Normalize）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. 读取训练集
train_dataset = datasets.ImageFolder(
    root="./data/train",
    transform=transform
)

# 3. 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False
)

# 4. 初始化统计变量
channel_sum = torch.zeros(3, dtype=torch.float64)
channel_sum_sq = torch.zeros(3, dtype=torch.float64)
num_pixels = 0

# 5. 遍历训练集
for images, labels in train_loader:
    images = images.double()

    # 计算各通道像素值之和
    channel_sum += images.sum(dim=(0, 2, 3))

    # 计算各通道像素值平方之和
    channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))

    # 统计每个通道的像素总数
    num_pixels += images.size(0) * images.size(2) * images.size(3)

# 6. 计算均值
# 均值等于各通道像素值之和除以像素总数
mean = channel_sum / num_pixels

# 7. 计算标准差
# 标准差等于各通道像素值平方之和除以像素总数减去均值平方的平方根
std = torch.sqrt(
    (channel_sum_sq / num_pixels - mean ** 2).clamp_min(0)
)

print("Mean:", mean.tolist())
print("Std:", std.tolist())
