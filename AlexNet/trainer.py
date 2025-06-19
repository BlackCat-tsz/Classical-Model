import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
from AlexNet import AlexNet
from config import Config
from MNIST_loader import get_dataloader

"""
    当前参数训练过后出现过拟合，在过拟合之前最高的准确率为
"""


if __name__ == '__main__':
    # 训练设备
    devices = utils.get_devices(Config.multi_gpu)
    print(f'训练轮次: {Config.epoch}')
    # 定义训练参数
    best_acc = 0.0
    early_stop = 0
    # 加载tensorboard
    writer = SummaryWriter(Config.tensorboard_path)
    # 检查模型存储路径是否存在
    model_path = utils.check_path(Config.model_path)
    # 加载数据、固定种子
    train_loader, test_loader = get_dataloader(Config.batch_size)
    # 固定随机种子
    utils.fix_seeds(Config.seed)

    # 定义模型、优化器、损失函数
    model = AlexNet(channel=Config.channel,
                    img_size=Config.img_size,
                    classes_num=Config.classes_num)
    # 将模型转移到多卡，实现多卡训练
    if Config.multi_gpu:
        model = nn.DataParallel(model, device_ids=devices)
    model.to(devices[0])
    loss_F = nn.CrossEntropyLoss().to(devices[0])  # 此交叉熵损失函数已经内含softmax，因此无需在模型中再显式添加
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.lr_scheduler)  # 余弦退火

    with tqdm(range(Config.epoch), desc='epoch', total=Config.epoch, ncols=80, position=0) as pbar:
        for epoch in pbar:

            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            len_train = 0
            len_test = 0

            # 开始训练
            model.train()
            for data in train_loader:
                len_train += Config.batch_size

                features, labels = data  # 读入数据和标记
                features, labels = features.to(devices[0]), labels.to(devices[0])

                optimizer.zero_grad()  # 梯度清零
                outputs = model(features)  # 前向传播
                loss = loss_F(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                # 取出outputs预测类别中的最大值，并计算正确的样本数量
                _, train_pred = torch.max(outputs, 1)
                train_acc += (train_pred.detach() == labels.detach()).sum().item()
                # 计算总损失
                train_loss += loss.item()

                # break

            # 开始验证
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    len_test += Config.batch_size

                    features, labels = data
                    features, labels = features.to(devices[0]), labels.to(devices[0])

                    outputs = model(features)
                    loss_val = loss_F(outputs, labels)  # 计算损失以供可视化

                    _, val_pred = torch.max(outputs, 1)
                    # get the index of the class with the highest probability
                    val_acc += (val_pred == labels).sum().item()
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
                torch.save(model.state_dict(), m_p)
                if model_path != Config.model_path:  # 移除之前最优mode.clpt
                    os.remove(model_path)
                model_path = m_p  # 更新路径和最佳准确率
                best_acc = acc

            early_stop += 1
            if early_stop > Config.early_stop:
                print("触发早停！")
            #     break
            scheduler.step()  # 更新学习率

    print(f"训练完成，best acc = {best_acc:.4f}")

    # 计算并打印参数量
    utils.count_parameters(model)

    writer.close()
