import torch
from torch import nn
from torchsummary import summary


# 残差块的构建,考虑有1*1的卷积的和没有1*1的卷积的
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,use_1conv, stride=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        # 如果输入的通道数和输出的通道数不相等，那么就需要用1*1的卷积进行转换,
        # 这里规定两个残差块的outchannel相同，即使输入通道为3，后面使用1*1的卷积将x通道数转换一下就可以相加了
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        # 这里的stride必须为1，因为第一个卷积可以决定是否要下采样，第二个卷积是为了提取特征，要保持通道数不变
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        if use_1conv:
            # 这里的stride也是参数传进来，是为了可以和第一个卷积的通道保持一致，方便相加
            self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None

        # 定义BN层
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    # 定义前向传播过程
    def forward(self,x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(x+y)
        return y
class ResNet18(nn.Module):
    # ResNet18继承nn.Module这个类，同时传入一个参数Residual，这里传是为了如果有不同的残差块都可以用，现在只有一个其实也可以不传入
    def __init__(self,Residual):
        # 调用父类的初始化函数
        super(ResNet18,self).__init__()
        # 定义块
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b2 = nn.Sequential(
            Residual(64,64,use_1conv=False,stride=1),
            Residual(64,64,use_1conv=False,stride=1)
        )

        self.b3 = nn.Sequential(
            Residual(64,128,use_1conv=True,stride=2),
            Residual(128,128,use_1conv=False,stride=1)
        )
        self.b4 = nn.Sequential(
            Residual(128,256,use_1conv=True,stride=2),
            Residual(256,256,use_1conv=False,stride=1)
        )
        self.b5 = nn.Sequential(
            Residual(256,512,use_1conv=True,stride=2),
            Residual(512,512,use_1conv=False,stride=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=512,out_features=2)
        )

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    print(summary(model,(3,224,224)))