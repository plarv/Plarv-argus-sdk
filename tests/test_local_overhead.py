import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plarv.local import LocalDetector

def run_benchmark(batch_time_ms: float, steps: int = 200, warmup: int = 50):
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    inputs = torch.randn(32, 512)
    targets = torch.randn(32, 512)
    
    # Baseline
    baseline_times = []
    for step in range(steps + warmup):
        start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_time_ms > 0:
            time.sleep(batch_time_ms / 1000.0)
            
        if step >= warmup:
            baseline_times.append(time.time() - start)
            
    baseline_avg_ms = (sum(baseline_times) / len(baseline_times)) * 1000.0

    # LocalDetector
    detector = LocalDetector(model, optimizer, silent=True)
    detector.attach()
    
    instrumented_times = []
    for step in range(steps + warmup):
        start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        detector.step()
        optimizer.step()
        
        if batch_time_ms > 0:
            time.sleep(batch_time_ms / 1000.0)
            
        if step >= warmup:
            instrumented_times.append(time.time() - start)
            
    detector.detach()
    
    instrumented_avg_ms = (sum(instrumented_times) / len(instrumented_times)) * 1000.0
    overhead_ms = instrumented_avg_ms - baseline_avg_ms
    overhead_pct = (overhead_ms / baseline_avg_ms) * 100 if baseline_avg_ms > 0 else 0
    
    print(f"Batch Time: {batch_time_ms:4}ms | Baseline: {baseline_avg_ms:6.2f}ms | With LocalDetector: {instrumented_avg_ms:6.2f}ms | Overhead: {overhead_ms:5.2f}ms ({overhead_pct:5.2f}%)")

if __name__ == "__main__":
    print("--- LocalDetector Overhead Benchmark ---")
    batch_times = [0, 10, 50, 100, 250, 500]
    for bt in batch_times:
        run_benchmark(batch_time_ms=bt)
