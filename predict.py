import torch
from PIL import Image
from torchvision import transforms
from model import GoogLeNet, Inception


def data_process(Image_path):
    # 读取本地的图片
    image = Image.open(Image_path)
    # 定义transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.488, 0.455, 0.417], std=[0.259, 0.252, 0.255])
    ])
    # 处理图片
    image = train_transform(image)
    image = image.unsqueeze(0)

    return image

# 预测函数
def predict(image):
    # 1.选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    # 2. 创建网络模型
    model = GoogLeNet(Inception).to(device)
    # 3. 加载训练好的模型权重
    model.load_state_dict(
        torch.load(
            "./best_model.pth",
            map_location=device
        )
    )
    # 4. 切换为评估模式
    model.eval()
    # 5. 进行预测
    # 10. 模型预测
    with torch.no_grad():
        outputs = model(image)

        # 转换为各类别概率
        probabilities = torch.softmax(outputs, dim=1)
        print(probabilities)
        # 获取概率最大的类别
        confidence, predicted = torch.max(
            probabilities, dim=1
        )
    # 11. 类别名称
    class_names = ["猫", "狗"]
    # 12. 输出预测结果
    result = class_names[predicted.item()]
    prob = confidence.item() * 100

    print(f"预测结果：{result}")
    print(f"预测概率：{prob:.2f}%")

if __name__ == "__main__":
    image = data_process("../GoogLeNet-catdog/local_test.png")
    predict(image)





