import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plarv.argus import Argus

def run_benchmark(batch_time_ms: float, steps: int = 100, warmup: int = 20):
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

    # Argus API
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "action": "CONTINUE",
            "harm_pressure": 0,
            "exists": True,
            "status": "Active"
        }).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        argus = Argus(
            api_key="test-key-overhead-1234",
            model=model,
            optimizer=optimizer,
            silent=True,
            fail_open=True
        )
        
        instrumented_times = []
        for step in range(steps + warmup):
            start = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            argus.step(loss=loss.item())
            optimizer.step()
            
            if batch_time_ms > 0:
                time.sleep(batch_time_ms / 1000.0)
                
            if step >= warmup:
                instrumented_times.append(time.time() - start)
                
        argus.complete()
    
    instrumented_avg_ms = (sum(instrumented_times) / len(instrumented_times)) * 1000.0
    overhead_ms = instrumented_avg_ms - baseline_avg_ms
    overhead_pct = (overhead_ms / baseline_avg_ms) * 100 if baseline_avg_ms > 0 else 0
    
    print(f"Batch Time: {batch_time_ms:4}ms | Baseline: {baseline_avg_ms:6.2f}ms | With Argus: {instrumented_avg_ms:6.2f}ms | Overhead: {overhead_ms:5.2f}ms ({overhead_pct:5.2f}%)")

if __name__ == "__main__":
    print("--- Argus API Overhead Benchmark ---")
    batch_times = [0, 10, 50, 100, 250, 500]
    for bt in batch_times:
        run_benchmark(batch_time_ms=bt)
