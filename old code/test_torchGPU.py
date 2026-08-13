import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
else:
    print("GPU 不可用")
    print("如果你的显卡是 RTX 5060，说明当前 PyTorch 版本不兼容。")