import warnings
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from ultralytics import RTDETR
from pathlib import Path

warnings.filterwarnings('ignore')

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

path = '/home/chuangxin/gaoruchen/detr-main/data/images/val/000a4bcdd.jpg'
model_path = Path('/home/chuangxin/gaoruchen/rtdetr-steel/weights/best.pt')
if __name__ == '__main__':
    model = RTDETR('../weights/best.pt')  # select your model.pt path
    results = model.predict(source=path,
                            conf=0.4,
                            # project='runs/detect',
                            # name='exp',
                            # save=True,
                            # visualize=True # visualize model features maps
                            # line_width=2, # line width of the bounding boxes
                            # show_conf=False, # do not show prediction confidence
                            # show_labels=False, # do not show prediction labels
                            # save_txt=True, # save results as .txt file
                            # save_crop=True, # save cropped images with results
                            )
    for r in results:
        print(r.boxes)
        