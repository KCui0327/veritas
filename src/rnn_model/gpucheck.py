# VENV name: cudaconfig
import torch

print("CUDA is available: ", torch.cuda.is_available())
# True

print("Number of CUDA devices: ", torch.cuda.device_count())
# 5

print("CUDA current device: ", torch.cuda.current_device())
# 0

print("CUDA device name: ", torch.cuda.get_device_name(0))

# 'NVIDIA GeForce RTX 3060'
print("CUDA version: ", torch.version.cuda)           # Should print a version like "12.1"
print("CUDNN version: ", torch.backends.cudnn.version())  # Should be an integer like 89

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))


import sys
print(sys.executable)