import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    cfg = "weights/best.pt"
    model = RTDETR(cfg)

    model.val(data='dataset/cfg/steel.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=8,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='rtdetr-ASFM-Steel',
              )