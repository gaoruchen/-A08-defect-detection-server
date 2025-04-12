import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
if __name__ == '__main__':
    device ="2"
    bs = 8  # 4 5G
    
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    model.train(data='dataset/cfg/FLIRv1_thermal_full_label.yaml',cache=False,imgsz=640,epochs=500,batch=bs,workers=1,device=device,project='runs/train',name='test-v',save_period=-1,)
'''
0 34619
1 
2 24413
3 
export CUDA_VISIBLE_DEVICES=0
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_d.py  \
> logs/nouse0.txt 2>&1 &
disown

export CUDA_VISIBLE_DEVICES=1
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_d.py  \
> logs/nouse1.txt 2>&1 &
disown

export CUDA_VISIBLE_DEVICES=2
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_d.py  \
> logs/nouse2.txt 2>&1 &
disown

export CUDA_VISIBLE_DEVICES=3
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_d.py  \
> logs/nouse3.txt 2>&1 &
disown
'''