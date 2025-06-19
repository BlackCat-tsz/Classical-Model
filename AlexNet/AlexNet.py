import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, channel, img_size, classes_num):
        super(AlexNet, self).__init__()

        # 计算展平后尺寸
        fla = img_size
        for i in range(3):
            fla = int((fla - 3) / 2) + 1
        fla = fla ** 2 * 256

        self.feature = nn.Sequential(
            nn.Conv2d(channel, 96, kernel_size=3, padding='same'),
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
            nn.Linear(fla, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 第二层
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 第三层
            nn.Linear(512, classes_num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.Connect(x)

        return x


# import utils
#
# if __name__ == '__main__':
#     model = AlexNet(1, 28, 10)
#
#     # 检测模型是否可运行
#     x = torch.ones(1, 1, 28, 28)
#     x = model(x)
#     print(x.size())
#
#     # 测试模型参数量
#     utils.count_parameters(model)
