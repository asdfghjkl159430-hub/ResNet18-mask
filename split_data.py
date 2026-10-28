
import os
import random
import shutil

# 原始数据集路径
data_dir = "./dataset"

# 划分后数据集保存路径
output_dir = "./data"

# 训练集比例
train_ratio = 0.8

# 固定随机种子，保证每次划分结果一致
random.seed(42)

# 猫狗类别
classes = ["mask", "no_mask"]

for class_name in classes:
    # 当前类别的原始路径
    class_dir = os.path.join(data_dir, class_name)

    # 获取当前类别所有图片名称
    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # 随机打乱图片
    images.sort()
    random.shuffle(images)

    # 计算训练集数量
    train_num = int(len(images) * train_ratio)

    # 划分训练集和测试集
    train_images = images[:train_num]
    test_images = images[train_num:]

    # 分别保存训练集和测试集
    for split, image_list in [
        ("train", train_images),
        ("test", test_images)
    ]:
        save_dir = os.path.join(
            output_dir, split, class_name
        )
        os.makedirs(save_dir, exist_ok=True)

        for image_name in image_list:
            src = os.path.join(class_dir, image_name)
            dst = os.path.join(save_dir, image_name)
            shutil.copy2(src, dst)

    print(
        f"{class_name}: "
        f"训练集 {len(train_images)} 张，"
        f"测试集 {len(test_images)} 张"
    )
