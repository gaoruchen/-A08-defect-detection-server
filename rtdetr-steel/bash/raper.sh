#!/bin/bash

# 基础配置
MEMORY_THRESHOLD=5000        # 显存阈值（单位：MB）
SLEEP_INTERVAL=300           # 轮询间隔（秒）
GPU_LIST=(0 2)          # 指定要抢占的GPU列表，例如 (0 1 2 3) 表示使用GPU0到GPU3
BATCH_SIZE=8                # 批量大小 4=5G 8=10G

#固定配置
CFG_NAME="rtdetr-r18"       # 固定的配置文件名称
PROJECT_NAME="test-v"       # 固定的输出路径名称

# 检查单个GPU显存剩余函数
check_single_gpu() {
    local gpu_id=$1
    local memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id)
    if [ "$memory_free" -gt $MEMORY_THRESHOLD ]; then
        return 0
    else
        return 1
    fi
}

# 启动训练任务函数
start_training() {
    local gpu_id=$1

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 启动训练任务: 配置=${CFG_NAME}, 使用GPU=${gpu_id}, 批量大小=${BATCH_SIZE}"

    # 运行训练脚本
    export CUDA_VISIBLE_DEVICES=$gpu_id
    /home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_parser.py \
        --cfg_name $CFG_NAME \
        --gpuid $gpu_id \
        --memo $PROJECT_NAME \
        --batch $BATCH_SIZE \
        --project runs/train \
        --name $PROJECT_NAME \
        --save_period -1 \
        > "logs/${PROJECT_NAME}-gpu${gpu_id}.log" 2>&1 &
    
    # 记录任务的PID
    local task_pid=$!
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 训练任务已启动，PID: $task_pid"
}

# 主循环
while true; do
    for i in "${!GPU_LIST[@]}"; do
        gpu_id="${GPU_LIST[$i]}"
        if check_single_gpu $gpu_id; then
            start_training $gpu_id
            # 任务启动后退出脚本
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 训练任务已在GPU${gpu_id}上启动，脚本结束。"
            exit 0
        fi
    done

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 本轮 GPU 检查结束，未找到可用GPU，休眠 ${SLEEP_INTERVAL} 秒。"
    sleep $SLEEP_INTERVAL
done