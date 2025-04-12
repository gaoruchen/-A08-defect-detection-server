import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    cfg_name = 'FRep/rtdetr-AIFI_Flash'
    model = RTDETR('mycfg/'+cfg_name+'.yaml')
    # model = RTDETR('ultralytics/cfg/models/rt-detr/'+cfg_name+'.yaml')

    model.train(data='dataset/cfg/NEU-DET.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                workers=8,
                device="1",
                # resume='',
                project='runs/train',
                name='NEU-DET_AIFI_Flash-',
                save_period=-1,
                )
'''
export CUDA_VISIBLE_DEVICES=1
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train1.py  \
> logs/FLIRv1-rtdetr-LCAF-CSWin_small-gpu1.log 2>&1 &
disown

rtdetr-AIFI_Flash
rtdetr-LCAF

dataset/cfg/LLVIP.yaml
dataset/cfg/FLIRv1_thermal_full_label.yaml
dataset/cfg/FLIRv2_thermal.yaml
dataset/cfg/FLIRv2_thermal_full_label
'''