
import os
import time
from quantml import Tensor, ops
from quantml.models import MLP

def get_line_count():
    total = 0
    for root, _, files in os.walk("."):
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                        total += len(file.readlines())
                except:
                    pass
    return total

def benchmark_mlp():
    model = MLP([100, 128, 128, 1])
    x = Tensor([[0.1] * 100])
    
    # Warmup
    for _ in range(10): model.forward(x)
    
    start = time.time()
    for _ in range(1000):
        model.forward(x)
    end = time.time()
    return (end - start) / 1000 * 1000 # ms

def test_deep_graph():
    # Verify autograd with deep graph (e.g. RNN unroll)
    x = Tensor([[0.1]], requires_grad=True)
    h = x
    depth = 50
    for _ in range(depth):
        h = ops.tanh(h)
    
    start = time.time()
    h.backward()
    end = time.time()
    return depth, (end - start) * 1000 # ms

if __name__ == "__main__":
    lines = get_line_count()
    mlp_latency = benchmark_mlp()
    depth, backprop_time = test_deep_graph()
    
    print(f"LINES_OF_CODE:{lines}")
    print(f"MLP_LATENCY_MS:{mlp_latency:.4f}")
    print(f"AUTOGRAD_DEPTH:{depth}")
    print(f"BACKPROP_TIME_MS:{backprop_time:.4f}")
