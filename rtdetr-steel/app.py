import torch
import cv2
import numpy as np
import warnings
import os
from ultralytics import RTDETR
from app.hm import get_model_hm
from flask import Flask, request, jsonify, Response


warnings.filterwarnings('ignore')

app = Flask(__name__)

# 类别标签
cls_label = {
    '0': '夹杂物',
    '1': '补丁',
    '2': '划痕',
    '3': '其他'
}

# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 构建模型路径
# model_path = os.path.join(current_dir, '..', 'weights', 'best.pt')
# # 使用绝对路径加载模型
# # 加载YOLO模型
# model = RTDETR(os.path.normpath(model_path))
# model_hm = get_model_hm()

model = RTDETR('weights/best.pt')
model_hm = get_model_hm()


def process_uploaded_file(file_stream):
    """处理上传的文件并返回解码后的图像"""
    try:
        img_bytes = file_stream.read()
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")
        return img, None
    except Exception as e:
        return None, str(e)


def perform_detection(image):
    """执行物体检测并返回结果"""
    torch.cuda.empty_cache()  # 清空缓存
    return model.predict(image, conf=0.395)  # imgsize=(256,1600)


@app.route('/api/python/detect', methods=['POST'])
def text_detection():
    """返回文本格式的检测结果"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img, error = process_uploaded_file(file)
    if error:
        return jsonify({'error': error}), 400

    try:
        results = perform_detection(img)
        detections = []
        for cls, conf in zip(
                results[0].boxes.cls.cpu(),
                results[0].boxes.conf.cpu()):
            detections.append({
                'label': cls_label[str(int(cls))],
                'confidence': round(float(conf), 3),
                # 'bbox': box.numpy().tolist()
            })
        return jsonify({'results': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/python/result_image', methods=['POST'])
def image_annotation():
    """返回标注后的JPEG图像"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img, error = process_uploaded_file(file)
    if error:
        return jsonify({'error': error}), 400

    try:
        results = perform_detection(img)
        plotted_img = results[0].plot()  # 假设返回RGB格式图像

        # 转换颜色空间并编码为JPEG
        plotted_img_bgr = cv2.cvtColor(plotted_img, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', plotted_img_bgr)

        return Response(img_encoded.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/python/heatmap', methods=['POST'])
def image_heatmap():
    """返回标注后的JPEG图像"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img, error = process_uploaded_file(file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if error:
        return jsonify({'error': error}), 400

    try:
        torch.cuda.empty_cache()  # 清空缓存

        try:
            img_hm = model_hm(img)
            if img_hm is None:
                raise ValueError('the steel is ok')
        except ValueError as e:
            return jsonify({'error': str(e)}), 400


        # 转换颜色空间并编码为JPEG
        heat_img_bgr = cv2.cvtColor(img_hm, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', heat_img_bgr)

        return Response(img_encoded.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
