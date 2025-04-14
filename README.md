# 车辆零部件缺陷检测系统服务端

本项目是基于 **Flask** 框架的车辆零部件缺陷检测系统的服务端（模型部署端）实现，以及基于 **PyTorch** 框架的模型训练实现。

---

## 项目功能

- **模型训练**  
  提供完整的模型训练代码，支持用户根据提供的数据集进行训练。
  
- **模型验证**  
  验证训练好的模型性能，评估其在测试集上的表现。

- **模型推理**  
  支持对单个或批量图片进行推理，快速检测车辆零部件是否存在缺陷。

- **部署 Flask 后端**  
  提供基于 Flask 的后端服务，支持通过 HTTP 请求调用模型进行推理。

---

## 安装依赖

确保项目运行在 **conda** 环境中，参考版本：`miniconda=4.12.0`，`python=3.8`。  
进入 conda 环境后执行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

---

## 运行项目

### 训练模型
需要从网盘下载划分好的数据集，数据集文件夹放在data目录下面，网盘链接：
https://pan.baidu.com/s/1061-tOXs9PRt6LnqylU4ng 提取码: k9vr

如果提示无法找到数据集，则把datasets/cfg/Mixture.yaml（steel.yaml和NEU-DET.yaml同理）里的相对路径改为绝对路径。

```bash
python train0.py
```

### 验证模型&测试mAP50
同样需要下载数据集

```bash
python val0.py
```

### 运行模型服务端
无需下载数据集

```bahs
python app.py
