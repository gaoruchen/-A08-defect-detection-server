import warnings
from ultralytics import RTDETR
warnings.filterwarnings('ignore')

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

path = '/home/chuangxin/gaoruchen/detr-main/data/images/val'
# path = '/home/chuangxin/gaoruchen/rtdetr-steel/data/NEU-DET/images/val'
if __name__ == '__main__':
    model = RTDETR('weights/best.pt')  # select your model.pt path
    model.predict(source=path,
                  conf=0.4,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )
