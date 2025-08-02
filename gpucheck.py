import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
logger.info(f"Current device: {torch.cuda.current_device()}")
logger.info(f"Device name: {torch.cuda.get_device_name(0)}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Step 2: Create a tensor and move it to the selected device
x = torch.rand(1000, 1000, device=device)
y = torch.rand(1000, 1000, device=device)

# Step 3: Perform a GPU-accelerated operation
z = torch.matmul(x, y)

# Step 4: Confirm where the result tensor is stored
if z.device.type == "cuda":
    logger.info(
        f"Operation performed on GPU: {torch.cuda.get_device_name(z.device.index)}"
    )
else:
    logger.info("Operation performed on CPU.")
