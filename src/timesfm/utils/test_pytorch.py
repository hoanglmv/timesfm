import torch
import timesfm

# Kiểm tra xem PyTorch có nhận diện được GPU không (nếu máy bạn có GPU)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Kiểm tra timesfm
print("TimesFM imported successfully!")