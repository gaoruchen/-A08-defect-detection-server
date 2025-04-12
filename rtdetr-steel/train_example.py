import warnings
import torch
import argparse
from ultralytics import RTDETR

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='RTDETR Training Script')
    
    # 添加命令行参数
    parser.add_argument('--cfg_name', type=str, default='rtdetr-LCAF',
                        help='configuration file name')
    parser.add_argument('--gpuid', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--memo', type=str, default='asfmgroups2_softplus',
                        help='additional memo for experiment')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建模型
    model = RTDETR('mycfg/' + args.cfg_name + '.yaml')
    
    # 训练模型
    model.train(
        data='dataset/cfg/FLIRv1_thermal_full_label.yaml',
        cache=False,
        imgsz=640,
        epochs=100,
        batch=4,
        workers=8,
        device=args.gpuid,
        # resume='',
        project='runs/train',
        name=f'flirv1-{args.cfg_name}-{args.memo}-e100-bs4-gpu{args.gpuid}-',
        save_period=-1,
    )

if __name__ == '__main__':
    main()