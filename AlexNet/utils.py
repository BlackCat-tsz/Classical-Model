import os
import torch
import numpy as np


# 固定随机种子
def fix_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 计算模型参数总量
def count_parameters(model):  # 计算参数总量
    print(f"当前模型参数总量约为：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


# 得到训练设备
def get_devices(multi_gpu):
    devices = []
    if torch.cuda.is_available():
        if multi_gpu:  # 多卡训练
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                devices.append('cuda:' + f'{i}')
        else:  # 单卡训练
            devices.append('cuda:0')
        print(f'{"使用多卡训练" if multi_gpu else "使用单卡训练"}，GPU可用，DEVICES: {devices}')
    else:  # GPU不可用
        devices.append('cpu')
        print(f'{"使用多卡训练" if multi_gpu else "使用单卡训练"}，但GPU不可用，DEVICES: {devices}')
    return devices


# 检查路径是否存在，不存在则建立对应路径
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# if __name__ == '__main__':
#     get_devices(False)
