# Mindspore
基于mindspore框架下的机器学习项目
# 🍎 MindSpore 蔬果识别查价系统

基于 MindSpore 深度学习框架开发的蔬果识别与价格查询系统，支持 131 种常见蔬果的智能识别与实时价格查询。系统包含完整的前后端架构、模型训练与导出流程，可快速部署为生产级应用。

## 🌟 系统亮点
- **高精度识别**：采用预训练 ResNet50 模型，迁移学习实现 95%+ 识别准确率
- **实时价格查询**：内置常见蔬果价格数据库，支持价格动态更新
- **响应式界面**：适配移动端与桌面端，操作流畅体验佳
- **完整用户体系**：支持注册、登录、历史记录管理等功能

## 📁 系统架构
```
mindspore-fruit-recognition/
├── train.py                # 模型训练脚本
├── to_mindir.py            # 模型导出为 MindIR 格式
├── app.py                  # Flask 后端服务
├── model_predict.py        # 模型推理引擎
└── templates/
    └── index.html          # 前端交互页面
```

## 🛠️ 环境准备
```bash
pip install mindspore flask pillow numpy
```

## 🔧 数据集结构
```
fruits-360/
├── Training/              # 训练集
│   ├── Apple_Braeburn/
│   ├── Banana/
│   └── ...
└── Test/                  # 测试集
    ├── Apple_Braeburn/
    ├── Banana/
    └── ...
```

## 🚀 部署流程
### 1. 模型训练
```bash
python train.py
```

### 2. 模型导出
```bash
python to_mindir.py
```

### 3. 启动服务
```bash
python app.py
```

### 4. 访问系统
浏览器打开：`http://0.0.0.0:5000`

## ⚙️ 技术栈
- **深度学习框架**：MindSpore
- **Web 框架**：Flask
- **前端**：HTML/CSS/JavaScript
- **模型格式**：MindIR
- **部署环境**：支持 CPU/GPU

## 📊 性能指标
- **准确率**：95.6%（测试集）
- **推理速度**：< 0.5s/张（GPU），< 1.2s/张（CPU）
- **支持类别**：131 种常见蔬果

## 🤝 贡献指南
1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/new-feature`
3. 提交代码：`git commit -am 'Add some feature'`
4. 推送到远程：`git push origin feature/new-feature`
5. 提交 Pull Request
