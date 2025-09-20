import torch

# 查看是否有可用的 CUDA 设备
print("CUDA available:", torch.cuda.is_available())

# 查看当前设备数量
print("Number of GPUs:", torch.cuda.device_count())

# 查看当前默认设备
print("Current GPU:", torch.cuda.current_device())

# 查看 GPU 名称
print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
