
import time
import pytest
from quantml import Tensor, ops
from quantml.models import MLP, LSTM, MultiHeadAttention

def benchmark_inference_speed():
    print("\n--- Inference Speed Benchmarks ---")
    
    # 1. MLP Inference
    model = MLP([100, 256, 256, 1])
    x = Tensor([[0.1] * 100]) # Batch size 1
    
    # Warmup
    for _ in range(10): model.forward(x)
    
    # Measure
    start = time.time()
    iters = 1000
    for _ in range(iters):
        model.forward(x)
    end = time.time()
    
    avg_latency = (end - start) / iters * 1000 # ms
    print(f"MLP (100->256->256->1) Latency: {avg_latency:.4f} ms")
    
    # 2. Attention Inference
    attn = MultiHeadAttention(embed_dim=64, num_heads=4)
    x_seq = Tensor([[[0.1]*64]*20]) # Batch 1, Seq 20, Dim 64
    
    # Warmup
    try:
        for _ in range(5): attn.forward(x_seq)
        
        start = time.time()
        iters = 100
        for _ in range(iters):
            attn.forward(x_seq)
        end = time.time()
        
        avg_latency = (end - start) / iters * 1000
        print(f"Self-Attention (Seq 20, Dim 64) Latency: {avg_latency:.4f} ms")
    except Exception as e:
        print(f"Attention Benchmark Failed: {e}")

def count_code_metrics():
    print("\n--- Code Metrics ---")
    import os
    
    total_lines = 0
    test_files = 0
    core_files = 0
    
    for root, _, files in os.walk("."):
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue
            
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as file:
                    try:
                        lines = len(file.readlines())
                        total_lines += lines
                        if "tests" in root:
                            test_files += 1
                        else:
                            core_files += 1
                    except:
                        pass
                        
    print(f"Total Python Lines: {total_lines}")
    print(f"Test Files: {test_files}")
    print(f"Core Files: {core_files}")

if __name__ == "__main__":
    benchmark_inference_speed()
    count_code_metrics()
