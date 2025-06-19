from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size):
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


# 测试数据大小
# if __name__ == '__main__':
#     train_loader, _ = get_dataloader(16)
#
#     for batch_idx, (images, labels) in enumerate(train_loader):
#         print(f"Batch{batch_idx + 1}")
#         print(f"图像张量形状：{images.shape}")
#         print(f"标签张量：{labels}")
#         break
