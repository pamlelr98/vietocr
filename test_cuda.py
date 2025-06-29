import torch
print(torch.__version__)
print(torch.cuda.is_available())
# Kết quả mong muốn là True
print(torch.cuda.get_device_name(0))
# Hiển thị tên GPU của bạn