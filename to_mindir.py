import os
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.train.serialization import load_checkpoint, export

def export_mindir_model():
    # ======================= 配置部分 =======================
    ckpt_path = "/root/ckpt/fruit_veg_recognition_3-10_2115.ckpt"  # 替换为你的ckpt路径
    output_name = "model3"               # 导出的MindIR文件名
    num_classes = 131                    # 当前任务的类别数
    input_shape = (1, 3, 224, 224)      # 模型输入形状
    # ======================================================

    # 设置运行环境
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    # ======================= 模型构建 =======================
    print("=== 构建模型 ===")
    from mindvision.classification.models import resnet50
    net = resnet50(pretrained=False, num_classes=1000)
    
    # 修改分类头
    in_features = net.head.dense.in_channels
    net.head.dense = nn.SequentialCell([
        nn.Dropout(keep_prob=0.5),
        nn.Dense(in_features, num_classes, weight_init='normal', bias_init='zeros')
    ])
    print(f"✅ 模型构建完成 | 分类层形状: {net.head.dense[1].weight.shape}")

    # ======================= 权重加载 =======================
    print(f"\n=== 加载检查点: {ckpt_path} ===")
    param_dict = load_checkpoint(ckpt_path)
    
    # 调试：打印检查点参数键名
    print("\n[调试] 检查点中的参数键名:")
    print(list(param_dict.keys()))

    # 自动检测分类层键名
    classifier_keys = [k for k in param_dict.keys() if any(x in k.lower() for x in ['dense', 'fc', 'head'])]
    if not classifier_keys:
        raise ValueError("未找到分类层参数！请检查检查点文件")
    
    weight_key = next(k for k in classifier_keys if 'weight' in k.lower())
    bias_key = next(k for k in classifier_keys if 'bias' in k.lower())
    print(f"检测到分类层参数: {weight_key}, {bias_key}")

    # 分类层权重适配
    old_classes = param_dict[weight_key].shape[0]
    if old_classes != num_classes:
        print(f"⚠️ 正在适配分类层权重 ({old_classes} -> {num_classes})...")
        
        # 保留原权重，新增随机初始化
        old_weight = param_dict[weight_key].asnumpy()
        old_bias = param_dict[bias_key].asnumpy()
        
        new_weight = np.random.normal(scale=0.01, size=(num_classes, old_weight.shape[1])).astype(np.float32)
        new_bias = np.zeros(num_classes, dtype=np.float32)
        
        # 保留原有权重
        copy_len = min(old_classes, num_classes)
        new_weight[:copy_len] = old_weight[:copy_len]
        new_bias[:copy_len] = old_bias[:copy_len]
        
        # 更新参数
        param_dict[weight_key] = ms.Parameter(ms.Tensor(new_weight))
        param_dict[bias_key] = ms.Parameter(ms.Tensor(new_bias))

    # 加载所有参数
    ms.load_param_into_net(net, param_dict)
    print("✅ 参数加载完成")

    # ======================= 模型导出 =======================
    print("\n=== 导出MindIR ===")
    dummy_input = Tensor(np.random.rand(*input_shape), dtype=ms.float32)
    export(
        net,
        dummy_input,
        file_name=output_name,
        file_format="MINDIR"
    )
    print(f"✅ 导出成功！文件已保存为: {output_name}.mindir")

if __name__ == "__main__":
    export_mindir_model()