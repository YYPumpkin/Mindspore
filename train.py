import os
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.dataset import transforms, vision
from mindspore.dataset import ImageFolderDataset
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Momentum
from mindspore.dataset.vision import Inter
from mindspore.amp import auto_mixed_precision 
from mindspore.nn import CosineDecayLR
from PIL import Image
import numpy as np

def main():
    print("join")
    # --- 1. 配置MindSpore运行环境 ---
    ms.set_seed(1) # 设置随机种子，保证每次运行结果的可复现性
    # 设置运行模式为图模式 (GRAPH_MODE)，通常性能更好
    # 设置设备目标为 "GPU" (如果你有GPU)，否则使用 "CPU"
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU") # <--- 重要：如果你没有GPU，请改为 "CPU"！

    # --- 2. 参数设置 ---
    batch_size = 32 # 每次训练的图片数量
    image_size = 224 # 模型期望的图片尺寸 (例如 ResNet 通常是 224x224)
    num_epochs = 10 # 训练的轮次 (增加到30轮，以获得更好的精度和收敛性)
    initial_learning_rate = 0.001 # 初始学习率，将通过调度器进行调整
    momentum = 0.9 # 动量参数，用于Momentum优化器
    weight_decay = 1e-4 # 权重衰减 (L2 正则化)，有助于防止过拟合，防止模型权重过大
    num_classes = 131 # <--- 请根据你的具体数据集修改此数值！

    # --- 3. 数据准备 ---
    def create_dataset(dataset_path, usage, batch_size, image_size, num_parallel_workers=8):
        # 将抽象的 'train'/'val' 映射到 Kaggle 数据集实际的文件夹名 'Training'/'Test'
        if usage == "train":
            actual_usage_folder = "Training"
        elif usage == "val":
            actual_usage_folder = "Test"
        else:
            raise ValueError("Usage 参数必须是 'train' 或 'val'。")

        dataset_folder = os.path.join(dataset_path, actual_usage_folder)
        # 检查数据集文件夹是否存在
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"数据集文件夹未找到: {dataset_folder}")

        # ImageFolderDataset 会自动根据文件夹名称识别类别和标签
        dataset = ImageFolderDataset(dataset_folder, num_parallel_workers=num_parallel_workers, shuffle=True)

        # 定义数据增强和预处理操作
        
        if usage == "train":
            resize_size_for_crop = 256
            transform_ops = transforms.Compose([
                # 更多样化的数据增强，提高模型泛化能力
                vision.Decode(),
                vision.Resize(image_size, interpolation=Inter.BICUBIC), # 调整图片大小
                vision.RandomCrop(image_size, padding=4),       # 随机裁剪
                vision.RandomHorizontalFlip(),                  # 随机水平翻转
                vision.RandomVerticalFlip(),                    # <--- 新增：随机垂直翻转
                vision.RandomRotation(degrees=(0, 30)),         # <--- 新增：随机旋转0到30度
                vision.RandomColorAdjust(brightness=(0.7, 1.3), # <--- 新增：色彩抖动
                                       contrast=(0.7, 1.3),
                                      saturation=(0.7, 1.3),
                                     hue=(-0.1, 0.1)),
                
                vision.Rescale(1.0 / 255.0, 0.0),               # 将像素值归一化到 [0, 1] 范围
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 使用ImageNet的均值和标准差进行标准化
                vision.HWC2CHW()                                # 将图片格式从 HWC (高、宽、通道) 转换为 CHW (通道、高、宽)
            ])
        else: # 验证集/测试集通常只进行裁剪和标准化，不进行随机增强
            transform_ops = transforms.Compose([
                vision.Decode(),
                vision.Resize(image_size, interpolation=Inter.BICUBIC),
                vision.CenterCrop(image_size),                  # 中心裁剪
                vision.Rescale(1.0 / 255.0, 0.0),
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ])

        dataset = dataset.map(operations=transform_ops, input_columns="image", num_parallel_workers=num_parallel_workers)
        dataset = dataset.batch(batch_size, drop_remainder=True) # drop_remainder=True 确保每个批次的大小固定
        return dataset

    print("--- 正在加载和准备数据集 ---")
    try:
    # 替换原来的class_names获取代码
        dataset_path = r"/root/.cache/fruits-360"
        training_dir = os.path.join(dataset_path, "Training")

        # 手动获取有效类别（子文件夹）
        class_names = [item for item in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, item))]
        num_classes = len(class_names)

        print(f"手动检测到的类别数量：{num_classes}")  # 应为206
        if num_classes == 0:
            print("错误：Training文件夹下没有有效子文件夹（类别）")
            exit()
        
        # 然后创建处理后的数据集
        train_dataset = create_dataset(dataset_path, "train", batch_size, image_size)
        val_dataset = create_dataset(dataset_path, "val", batch_size, image_size)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请检查 KaggleHub 下载的数据集结构是否符合预期，即在下载目录的 'fruits-360' 文件夹内有 'Training' 和 'Test' 文件夹。")
        exit() # 如果找不到数据集，则退出程序
    print(f"检测到的类别: {class_names}")
    if len(class_names) != num_classes:
        print(f"⚠️ 警告: 设置的 'num_classes' 参数 ({num_classes}) 与实际检测到的类别数量 ({len(class_names)}) 不匹配。")
        print(f"已自动将 num_classes 设置为实际检测到的数量: {len(class_names)}。")
        num_classes = len(class_names)

    # --- 4. 定义网络 (使用预训练的ResNet50进行迁移学习) ---
    print("--- 正在定义网络 ---")
    try:
        # mindvision 提供了方便的预训练模型。通过 'pip install mindvision' 安装。
        from mindvision.classification.models import resnet50
        # 加载预训练的ResNet50模型。'pretrained=True' 会下载 ImageNet 权重。
        net = resnet50(pretrained=True, num_classes=1000) # 原始 ImageNet 数据集有 1000 个类别
    except ImportError:
        print("❌ 无法从 mindvision.classification.models 导入 'resnet50'。")
        print("请确保已安装 'mindvision' ('pip install mindvision')。")
        print("作为替代，这里将构建一个简化的ResNet骨干，但它不是预训练的，性能会差很多。")
        # 如果无法导入mindvision，提供一个简化的ResNet骨干作为备用（非预训练）
        from mindspore.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d
        from mindspore.common.initializer import Normal
        
        # 定义残差块
        class ResidualBlock(nn.Cell):
            expansion = 4
            def __init__(self, in_channels, out_channels, stride=1):
                super(ResidualBlock, self).__init__()
                self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=False)
                self.bn1 = BatchNorm2d(out_channels)
                self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, has_bias=False, pad_mode='pad')
                self.bn2 = BatchNorm2d(out_channels)
                self.conv3 = Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, has_bias=False)
                self.bn3 = BatchNorm2d(out_channels * self.expansion)
                self.relu = ReLU()

                self.shortcut = nn.SequentialCell()
                if stride != 1 or in_channels != out_channels * self.expansion:
                    self.shortcut = nn.SequentialCell([
                        Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, has_bias=False),
                        BatchNorm2d(out_channels * self.expansion)
                    ])

            def construct(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                out += self.shortcut(identity)
                out = self.relu(out)
                return out

        # 定义ResNet主干网络
        class ResNet(nn.Cell):
            def __init__(self, block, layers, num_classes=1000):
                super(ResNet, self).__init__()
                self.in_channels = 64
                self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode='pad')
                self.bn1 = BatchNorm2d(64)
                self.relu = ReLU()
                self.maxpool = MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

                self.avgpool = AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Dense(512 * block.expansion, num_classes, weight_init=Normal(0.01))

            def _make_layer(self, block, out_channels, blocks, stride=1):
                layers = []
                layers.append(block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.in_channels, out_channels, 1))
                return nn.SequentialCell(layers)

            def construct(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = x.view(x.shape[0], -1) # 展平
                x = self.fc(x)
                return x
        net = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=1000) # ResNet50 的层数配置
    # 获取原始 ResNet 全连接层的输入特征数
    in_features = net.head.dense.in_channels
    # 将 ResNet 的全连接层替换为适合新类别数量的层
    # 添加一个 Dropout 层以进一步正则化，防止过拟合
    net.head.dense = nn.SequentialCell([
        nn.Dropout(keep_prob=0.5), # <--- 新增：Dropout 层，保留 50% 的神经元
        nn.Dense(in_features, num_classes, weight_init='normal', bias_init='zeros')
    ])

    # 推荐：启用混合精度训练。这可以在支持FP16的GPU上显著加速训练并减少显存占用。
    #net = auto_mixed_precision(net, 'O2') # 'O2' 是一个常用的优化级别

    # --- 5. 定义损失函数和优化器 ---
    print("--- 正在定义损失函数和优化器 ---")
    # 损失函数：用于多分类任务的交叉熵损失，sparse=True 表示标签是整数索引
    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义学习率调度器：余弦退火学习率 (Cosine Annealing LR)
    # steps_per_epoch 是一个 epoch 中的批次数量
    steps_per_epoch = train_dataset.get_dataset_size()
    total_steps = num_epochs * steps_per_epoch
    lr_scheduler = CosineDecayLR(min_lr=1e-6, max_lr=initial_learning_rate, decay_steps=total_steps)
    print(f"总训练步数 (steps): {total_steps}")
    print(f"初始学习率: {initial_learning_rate}, 最小学习率: 1e-4")

    # 优化器：使用 Momentum 优化器更新网络参数，并传入学习率调度器和权重衰减
    optimizer = Momentum(net.trainable_params(), lr_scheduler, momentum, weight_decay=weight_decay)

    # --- 6. 训练模型 ---
    print("--- 开始训练模型 ---")
    # 实例化模型对象，传入网络、损失函数、优化器和评估指标
    model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})

    # 设置回调函数：保存模型检查点、监控训练损失和记录 Summary 日志
    # ckpt 文件夹用于保存模型权重
    ckpt_dir = "./ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5) # 每个 epoch 结束保存一次检查点
    ckpoint_cb = ModelCheckpoint(prefix="fruit_veg_recognition", directory=ckpt_dir, config=config_ck)
    # LossMonitor：在训练过程中打印损失信息
    loss_monitor_cb = LossMonitor(per_print_times=1) # 每隔1步打印一次损失

    # SummaryCollector：用于 MindInsight 可视化，日志会保存到 summary_log 文件夹
    summary_dir = "./summary_log"
    os.makedirs(summary_dir, exist_ok=True)
    # collect_freq=1 表示每隔1步收集一次数据。可以适当调大以减少日志文件大小。
    summary_collector_cb = SummaryCollector(summary_dir=summary_dir, collect_freq=1)

    # 启动模型训练
    model.train(num_epochs, train_dataset, callbacks=[ckpoint_cb, loss_monitor_cb, summary_collector_cb], dataset_sink_mode=True)

    print("--- 模型训练完成 ---")
    print(f"训练日志已保存到 '{summary_dir}' 目录。您可以通过运行 'mindinsight start --summary-base-dir {summary_dir}' 命令来查看。")
    print(f"模型检查点已保存到 '{ckpt_dir}' 目录。")

    # --- 7. 评估模型 ---
    print("--- 开始评估模型 ---")
    metric = model.eval(val_dataset)
    print(f"验证集精度: {metric['accuracy']:.4f}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    print("结束训练")