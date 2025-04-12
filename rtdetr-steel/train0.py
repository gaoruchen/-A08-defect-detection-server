import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    cfg_name = 'rtdetr-r18'
    # model = RTDETR('mycfg/LCAF/'+cfg_name+'.yaml')
    model = RTDETR('mycfg/rtdetr-r18.yaml')
    # model = RTDETR('weights/best.pt')
    # model = RTDETR('ultralytics/cfg/models/rt-detr/'+cfg_name+'.yaml')

    model.train(data='dataset/cfg/Mixture.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=8,
                workers=8,
                # device="0",
                # resume='',
                project='runs/train',
                name=cfg_name+'-Mixture-',
                save_period=-1,
                )
'''
export CUDA_VISIBLE_DEVICES=0
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train0.py  \
> logs/FLIRv1-rtdetr-AIFI_Flash1-fasternet_t2.log 2>&1 &
disown

rtdetr-AIFI_Flash
rtdetr-LCAF

dataset/cfg/LLVIP.yaml
dataset/cfg/FLIRv1_thermal_full_label.yaml
dataset/cfg/FLIRv2_thermal.yaml
dataset/cfg/FLIRv2_thermal_full_label
'''