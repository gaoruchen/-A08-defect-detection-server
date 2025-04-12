#!/bin/bash

# 基础配置
MEMORY_THRESHOLD=8000        # 显存阈值
SLEEP_INTERVAL=300          # 轮询间隔(秒)
GPU_LIST=(0 1 2 3)             # 指定要使用的GPU列表，例如(2 3)表示只使用GPU2和GPU3
MAX_TASKS_PER_GPU=3       # 每张GPU上最多运行的rtdetr任务数量

# 定义训练配置列表
declare -a CONFIG_LIST=(

"rtdetr-AIFI_Flash-C2f-MSMHSA-CGLU:my_memo"
"rtdetr-AIFI_Flash-C2f-CAMixer:my_memo"
"rtdetr-AIFI_Flash-C2f-AddutuveBlock-CGLU:my_memo"
"rtdetr-AIFI_Flash-C2f-RAB:my_memo"
"rtdetr-AIFI_Flash-C2f-SHSA-CGLU:my_memo"
"rtdetr-AIFI_Flash-Man-Star:my_memo"
"rtdetr-AIFI_Flash-gConvC3:my_memo"
"rtdetr-AIFI_Flash-CSP-PMSFA:my_memo"

)

# 当前配置索引
CURRENT_CONFIG=0

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

# 检查GPU上运行的rtdetr任务数量
check_gpu_tasks() {
    local gpu_id=$1
    local task_count=0

    # 使用nvidia-smi获取当前GPU上运行的rtdetr任务数量
    task_count=$(nvidia-smi -i $gpu_id --query-compute-apps=pid,process_name --format=csv,noheader | grep "rtdetr" | wc -l)

    if [ $task_count -lt $MAX_TASKS_PER_GPU ]; then
        return 0
    else
        return 1
    fi
}

# 启动训练任务函数
start_training() {
    local gpu_id=$1
    local config_pair=${CONFIG_LIST[$CURRENT_CONFIG]}
    local CFG_NAME=${config_pair%%:*}
    local MEMO=${config_pair#*:}

    # 创建日志目录
    mkdir -p logs

    # 构建日志文件名
    local log_file="logs/FLIR-${CFG_NAME}-${MEMO}-gpu${gpu_id}.log"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 启动训练任务: 配置=${CFG_NAME}, 备注=${MEMO}, 使用GPU=${gpu_id}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 日志文件: ${log_file}"

    # 运行训练脚本
    export CUDA_VISIBLE_DEVICES=$gpu_id
    /home/lihan/miniconda3/envs/rtdetr/bin/python /home/lihan/Code/rtdetr/train_parser.py \
        --cfg_name $CFG_NAME \
        --gpuid $gpu_id \
        --memo $MEMO \
        > "$log_file" 2>&1 &
    
    # 记录任务的PID
    local task_pid=$!
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 训练任务已启动，PID: $task_pid"

    # 更新配置索引
    CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
}

# 主循环
while true; do
    if [ $CURRENT_CONFIG -ge ${#CONFIG_LIST[@]} ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 配置列表已全部运行完毕，脚本结束。"
        break
    fi

    for i in "${!GPU_LIST[@]}"; do
        gpu_id="${GPU_LIST[$i]}"
        if check_single_gpu $gpu_id && check_gpu_tasks $gpu_id; then
            start_training $gpu_id
            # 检查是否所有配置已运行完毕
            if [ $CURRENT_CONFIG -ge ${#CONFIG_LIST[@]} ]; then
                echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 配置列表已全部运行完毕，脚本结束。"
                exit 0
            fi
        fi
    done

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] 本轮 GPU 检查结束，休眠 ${SLEEP_INTERVAL} 秒。"
    sleep $SLEEP_INTERVAL
done