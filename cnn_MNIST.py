import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

"""
    当前参数训练过后出现过拟合，在过拟合之前最高的准确率为99.19%
"""


class Config:
    # 输入图像尺寸
    channel = 1
    img_size = 28
    fla = 1024

    # n分类
    target_num = 10

    # 训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 超参数
    epoch = 1
    lr = 1e-4
    weight_decay = 1e-3  # L2正则化系数
    seed = 519
    batch_size = 64

    # 地址
    tensorboard_path = "./logs/batch_64"
    model_path = "./model/batch_64"

    # 早停
    early_stop = 10


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_Data(batch_size):
    # 定义加载时张量处理函数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3801,))
    ])

    # 导入数据集
    train_dataset = datasets.MNIST(
        root='./data/mnist',  # 数据集存储路径
        train=True,  # True=训练集，False=测试集
        download=True,  # 若本地无数据则自动下载
        transform=transform
    )
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

    # 创建数据加载器
    # batch_size:设置一次训练的数量
    # shuffle: 是否打乱数据,true为打乱
    # num_workers: 是否多进程加载数据（windows系统下可能会出现问题）
    # drop_lass: 训练取剩数据是否舍去，true为舍去
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def set_config(train_loader):
    # DataLoader中有自定义的迭代器，每次遍历返回一个batch 四维张量：[batch size, channels, width, height]
    # 下面是两种不同的提取方法
    for data in train_loader:
        imgs, targets = data
        shape = imgs.shape
        Config.channel, Config.img_size = shape[1], shape[2]  # 提取图片的通道数、宽、高
        # 计算展平后尺寸
        Config.fla = Config.img_size
        for i in range(3):
            Config.fla = int((Config.fla - 3) / 2) + 1
        Config.fla = Config.fla ** 2 * 256

        # print(config.channel, config.img_h, config.img_w)
        # print(imgs.shape)
        # print(targets)
        break


# for batch_idx, (images, labels) in enumerate(train_loader):
#     print(f"Batch{batch_idx + 1}")
#     print(f"图像张量形状：{images.shape}")
#     print(f"标签张量：{labels}")
#     break

class Cnn(nn.Module):
    def __init__(self, config):
        super(Cnn, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(config.channel, 96, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.Connect = nn.Sequential(
            # 第一层
            nn.Linear(config.fla, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 第二层
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 第三层
            nn.Linear(512, config.target_num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.Connect(x)

        return x


def count_parameters(model):  # 计算参数总量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print(f'训练设备: {Config.device}\n'
          f'训练轮次: {Config.epoch}')
    best_acc = 0.0
    writer = SummaryWriter(Config.tensorboard_path)
    early_stop = 0

    # 检查模型存储路径是否存在
    if not os.path.exists(Config.model_path):
        os.makedirs(Config.model_path)
    model_path = Config.model_path

    # 加载数据、固定种子
    train_loader, test_loader = get_Data(Config.batch_size)
    set_config(train_loader)
    same_seeds(Config.seed)

    # 定义模型、优化器、损失函数
    cnn = Cnn(Config)
    cnn = cnn.to(Config.device)
    loss_F = nn.CrossEntropyLoss()  # 此交叉熵损失函数已经内含softmax，因此无需在模型中再显式添加
    loss_F = loss_F.to(Config.device)
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 余弦退火


    with tqdm(range(Config.epoch), desc='epoch', total=Config.epoch, ncols=80, position=0) as pbar:
        for epoch in pbar:
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            len_train = 0
            len_test = 0

            # 开始训练
            cnn.train()
            for data in train_loader:
                len_train += Config.batch_size

                imgs, targets = data  # 读入数据和标记
                imgs, targets = imgs.to(Config.device), targets.to(Config.device)

                optimizer.zero_grad()  # 梯度清零
                outputs = cnn(imgs)  # 前向传播
                loss = loss_F(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                # 取出outputs预测类别中的最大值，并计算正确的样本数量
                _, train_pred = torch.max(outputs, 1)
                train_acc += (train_pred.detach() == targets.detach()).sum().item()
                # 计算总损失
                train_loss += loss.item()

                # break

            # 开始验证
            cnn.eval()
            with torch.no_grad():
                for data in test_loader:
                    len_test += Config.batch_size

                    imgs, targets = data
                    imgs, targets = imgs.to(Config.device), targets.to(Config.device)

                    outputs = cnn(imgs)
                    loss_val = loss_F(outputs, targets)  # 计算损失以供可视化

                    _, val_pred = torch.max(outputs, 1)
                    # get the index of the class with the highest probability
                    val_acc += (val_pred == targets).sum().item()
                    val_loss += loss_val.item()
                    # break

            writer.add_scalar("Loss/train", train_loss / len_train, epoch)
            writer.add_scalar("Loss/valid", val_loss / len_test, epoch)
            writer.add_scalar("Acc/train", train_acc / len_train, epoch)
            writer.add_scalar("Acc/valid", val_acc / len_test, epoch)

            acc = val_acc / len_test
            if acc > best_acc:
                early_stop = 0
                model_name = f'acc{acc:.4f}.ckpt'
                m_p = os.path.join(Config.model_path, model_name)
                torch.save(cnn.state_dict(), m_p)
                if model_path != Config.model_path:  # 移除之前最优mode.clpt
                    os.remove(model_path)
                model_path = m_p  # 更新路径和最佳准确率
                best_acc = acc

            early_stop += 1
            if early_stop > Config.early_stop:
                print("触发早停！")
                break
            scheduler.step()  # 更新学习率

    print(f"训练完成，best acc = {best_acc:.4f}")
    # 计算并打印参数量
    total_params = count_parameters(cnn)
    print(f"当前模型总参数量: {total_params:,} (约 {total_params / 1e6:.2f}M)")

    writer.close()
