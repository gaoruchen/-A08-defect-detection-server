import warnings
import torch
import argparse
from ultralytics import RTDETR

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='RTDETR Training Script')
    
    # 添加命令行参数
    parser.add_argument('--cfg_name', type=str, default='rtdetr-r18',
                        help='configuration file name')
    parser.add_argument('--gpuid', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--memo', type=str, default='memo',
                        help='additional memo for experiment')
    parser.add_argument('--batch', type=int, default=4,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs for training')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='image size for training')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of workers for data loading')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='project name for saving results')
    parser.add_argument('--save_period', type=int, default=-1,
                        help='save checkpoint every N epochs (-1 to disable)')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建模型
    # model = RTDETR(f'ultralytics/cfg/models/rt-detr/{args.cfg_name}.yaml')
    model = RTDETR('mycfg/' + args.cfg_name + '.yaml')
    
    # 训练模型
    model.train(
        # data='dataset/cfg/LLVIP.yaml',
        data='dataset/cfg/FLIRv1_thermal_full_label.yaml',  # 数据集配置文件
        cache=False,  # 是否缓存数据集
        imgsz=args.imgsz,  # 图像大小
        epochs=args.epochs,  # 训练轮数
        batch=args.batch,  # 批量大小
        workers=args.workers,  # 数据加载线程数
        device=args.gpuid,  # 使用的GPU ID
        project=args.project,  # 保存结果的目录
        name=f'FLIRv1-{args.cfg_name}-{args.memo}-e100-bs4-gpu{args.gpuid}-',# 实验名称
        save_period=args.save_period,  # 保存模型的频率
    )

if __name__ == '__main__':
    main()