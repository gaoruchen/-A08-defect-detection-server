import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    cfg_name = 'FRep/rtdetr-AIFI_Flash'
    model = RTDETR('mycfg/'+cfg_name+'.yaml')
    model.train(data='dataset/cfg/Mixture.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                workers=8,
                # device="2",
                # resume='',
                project='runs/train',
                name='Mixture_AIFI_Flash-',
                save_period=-1,
                )
'''
export CUDA_VISIBLE_DEVICES=2
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train2.py  \
> logs/FLIRv1-rtdetr-LCAF-r34-gpu2.log 2>&1 &
disown

rtdetr-AIFI_Flash
rtdetr-LCAF

dataset/cfg/LLVIP.yaml
dataset/cfg/FLIRv1_thermal_full_label.yaml
dataset/cfg/FLIRv2_thermal.yaml
dataset/cfg/FLIRv2_thermal_full_label
'''