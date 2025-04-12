import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    

    cfg_name = 'rtdetr-r18'
   
    model = RTDETR('mycfg/'+cfg_name+'.yaml')
    # model = RTDETR('ultralytics/cfg/models/rt-detr/'+cfg_name+'.yaml')
    model.train(data='dataset/cfg/FLIRv1_thermal_full_label.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=4,
                workers=8,
                device="cpu",
                # resume='',
                project='runs/train',
                name='test-v',
                save_period=-1,
                )
'''
nohup \
/home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_test.py  \
> logs/test.log 2>&1 &
disown

'''