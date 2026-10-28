
import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import ResNet18,Residual
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder


def train_val_data_process():
    # 定义数据集的路径
    ROOT_PATH = r'../ResNet-18/data/train'
    # 计算均值和方差，并进行标准化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.549, 0.502, 0.481], std=[0.318, 0.314, 0.318])
    ])
    train_data = ImageFolder(root=ROOT_PATH, transform=train_transform)


    train_data, val_data = torch.utils.data.random_split(
        train_data,
        [round(0.8 * len(train_data)),
         round(0.2 * len(train_data))]
    )

    train_dataloader = Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = Data.DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, val_dataloader


def model_train_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练设备: {device}")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -1.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print("-" * 50)

        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        # ==================== 训练阶段 ====================
        model.train()

        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1} [Train]",
            unit="batch",
            dynamic_ncols=True
        )

        for step, (b_x, b_y) in enumerate(train_bar):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 前向传播
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计当前训练情况
            batch_size = b_x.size(0)
            train_loss += loss.item() * batch_size
            train_corrects += (pre_lab == b_y).sum().item()
            train_num += batch_size

            # 实时更新进度条
            train_bar.set_postfix({
                "Loss": f"{train_loss / train_num:.4f}",
                "Acc": f"{train_corrects / train_num:.2%}"
            })

        # ==================== 验证阶段 ====================
        model.eval()

        val_bar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch + 1} [Val]  ",
            unit="batch",
            dynamic_ncols=True
        )

        # 验证时不计算梯度
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_bar):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                batch_size = b_x.size(0)
                val_loss += loss.item() * batch_size
                val_corrects += (pre_lab == b_y).sum().item()
                val_num += batch_size

                # 实时更新验证进度条
                val_bar.set_postfix({
                    "Loss": f"{val_loss / val_num:.4f}",
                    "Acc": f"{val_corrects / val_num:.2%}"
                })

        # ==================== 保存训练指标 ====================
        epoch_train_loss = train_loss / train_num
        epoch_train_acc = train_corrects / train_num
        epoch_val_loss = val_loss / val_num
        epoch_val_acc = val_corrects / val_num

        train_loss_all.append(epoch_train_loss)
        train_acc_all.append(epoch_train_acc)
        val_loss_all.append(epoch_val_loss)
        val_acc_all.append(epoch_val_acc)

        print(
            f"Train Loss: {epoch_train_loss:.4f} "
            f"Train Acc: {epoch_train_acc:.4f}"
        )
        print(
            f"Val Loss:   {epoch_val_loss:.4f} "
            f"Val Acc:   {epoch_val_acc:.4f}"
        )

        # 保存验证准确率最高的模型参数
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"发现更优模型！Best Val Acc: {best_acc:.4f}")

        # 计算累计训练时间
        time_use = time.time() - since
        print(
            f"累计耗时: {time_use // 60:.0f}m "
            f"{time_use % 60:.0f}s"
        )

    # ==================== 训练结束 ====================
    torch.save(best_model_wts, '../ResNet-18/best_model.pth')

    print("\n训练完成！")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print("最佳模型已保存至 best_model.pth")

    train_process = pd.DataFrame({
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'],
             train_process.train_loss_all,
             "ro-", label="Train loss")
    plt.plot(train_process['epoch'],
             train_process.val_loss_all,
             "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'],
             train_process.train_acc_all,
             "ro-", label="Train acc")
    plt.plot(train_process['epoch'],
             train_process.val_acc_all,
             "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    model = ResNet18(Residual)

    train_dataloader, val_dataloader = train_val_data_process()

    train_process = model_train_process(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=30
    )

    matplot_acc_loss(train_process)
