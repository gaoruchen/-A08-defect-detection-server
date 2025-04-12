import torch
if torch.cuda.is_available():
    torch.cuda.set_device(2)  # 选择 GPU 2
    torch.cuda.empty_cache()  # 释放显存
    print("GPU 2 显存已释放")
else:
    print("CUDA 不可用")