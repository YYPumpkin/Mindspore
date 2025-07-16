import os
import base64
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
from model_predict import ModelPredictor
from mindspore.dataset import vision, transforms
import mindspore as ms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 限制上传5MB

# 初始化模型，使用 ModelPredictor1 类
predictor = ModelPredictor(r"/root/model3.mindir")

def preprocess_test_image(image, image_size=224):
    """测试集预处理（与验证集严格一致）"""
    try:
        # 将PIL图像转换为numpy数组（HWC格式，RGB顺序）
        img_array = np.array(image)
        
        # 定义测试集预处理流水线（与验证集相同），但不使用vision.Decode()
        transform_ops = transforms.Compose([
            vision.Resize(image_size, interpolation=vision.Inter.BICUBIC),
            vision.CenterCrop(image_size),  # 中心裁剪
            vision.Rescale(1.0 / 255.0, 0.0),  # 归一化到 [0,1]
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
            vision.HWC2CHW()  # 转换通道顺序
        ])
        
        # 应用预处理
        processed_img = transform_ops(img_array)
        
        # 添加批处理维度 (1, C, H, W)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # 转换为MindSpore张量
        processed_img = ms.Tensor(processed_img, dtype=ms.float32)
        
        return processed_img  # 返回张量（BCHW格式，已含标准化）
    
    except Exception as e:
        raise RuntimeError(f"测试图片预处理失败: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        if 'image' not in request.files:
            return "没有上传文件", 400
            
        file = request.files['image']
        if file.filename == '':
            return "未选择文件", 400

        try:
            # 读取图片
            img = Image.open(BytesIO(file.read()))
            
            # 确保图像是RGB格式
            img = img.convert('RGB')
            
            # 预处理并预测
            input_data = preprocess_test_image(img)
            prediction = predictor.predict(input_data.asnumpy())
            
            # 打印预测结果以便调试
            print(f"原始预测结果: {prediction}")
            
            # 生成结果可视化 
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # 确保结果格式正确
            if isinstance(prediction, dict):
                if "class_name" in prediction and "confidence" in prediction and "price" in prediction:
                    result = {
                        "class_name": prediction["class_name"],
                        "confidence": prediction["confidence"],
                        "price": prediction["price"],
                        "visual": img_base64
                    }
                else:
                    result = {"error": f"预测结果缺少必要字段: {prediction}"}
            else:
                result = {"error": f"预测结果格式不正确: {type(prediction)}"}
                
        except Exception as e:
            result = {"error": f"预测过程出错: {str(e)}"}
            # 打印详细错误信息到控制台
            import traceback
            print(traceback.format_exc())

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)