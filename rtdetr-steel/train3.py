import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    cfg_name = 'LCAF/rtdetr-ASFM'
    # model = RTDETR('mycfg/'+cfg_name+'.yaml')
    model = RTDETR('weights/best.pt')
    
    model.train(data='dataset/cfg/Mixture.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=8,
                workers=8,
                # device="3",
                # resume='',
                project='runs/train',
                name='Mixture-'+cfg_name+'-weitiao-',
                save_period=-1,

                freeze=8,  # 冻结backbone
                pretrained=True,  # 使用预训练模型
                classes=[1, 3, 4]  # 重点类别
                )
'''
export CUDA_VISIBLE_DEVICES=3
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train3.py  \
> logs/FLIRv1-rtdetr-AIFI_Flash-convnextv2_nano.log 2>&1 &
disown

rtdetr-AIFI_Flash
rtdetr-LCAF

dataset/cfg/LLVIP.yaml
dataset/cfg/FLIRv1_thermal_full_label.yaml
dataset/cfg/FLIRv2_thermal.yaml
dataset/cfg/FLIRv2_thermal_full_label
'''