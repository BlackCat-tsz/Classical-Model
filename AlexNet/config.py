class Config:
    # 输入图像尺寸
    channel = 1
    img_size = 28

    # n分类
    classes_num = 10

    # 超参数
    epoch = 100
    lr_scheduler = 20
    lr = 1e-4
    weight_decay = 1e-3  # L2正则化系数
    seed = 519
    batch_size = 16

    # 地址
    tensorboard_path = "./logs"
    model_path = "./model"

    # 早停
    early_stop = 10

    # 多卡训练
    multi_gpu = False
