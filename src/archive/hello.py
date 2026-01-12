import torch
import time

print("------------------------------------------")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name:        {gpu_name}")
    
    # 5090の計算テスト（行列演算）
    print("\nStarting Tensor Calculation Test...")
    # 負荷をかけてみる（10000x10000の行列計算）
    x = torch.rand(10000, 10000, device=device)
    y = torch.rand(10000, 10000, device=device)
    
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize() # 計算終了を待つ
    end = time.time()
    
    print(f"Calculation Done! Time: {end - start:.4f} sec")
    print("RTX 5090 is READY to serve.")
else:
    print("GPU not detected...")
print("------------------------------------------")