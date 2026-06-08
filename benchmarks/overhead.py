import time
import sys
import os
import statistics
from unittest.mock import patch

# Add the current directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plarv.argus import Argus

def run_benchmark(batch_time_ms, num_steps=1000):
    # Initialize Argus in silent mode
    argus = Argus(api_key="bench-key", silent=True)
    
    # 🛡️ SOVEREIGN MOCKING: prevent real network calls during handshake/telemetry
    with patch('plarv.argus._post') as mock_post:
        mock_post.return_value = {"action": "CONTINUE", "harm_pressure": 0}
        argus._fire_async = lambda x: None
        
        overheads = []
        
        for i in range(num_steps):
            # Simulate batch processing
            time.sleep(batch_time_ms / 1000.0)
            
            # Measure overhead of argus.step()
            start = time.perf_counter()
            argus.step(loss=0.5)
            end = time.perf_counter()
            
            overheads.append((end - start) * 1000) # ms
        
    avg_overhead = statistics.mean(overheads)
    p95_overhead = sorted(overheads)[int(len(overheads)*0.95)]
    
    # Impact = overhead / (batch_time + overhead)
    impact_pct = (avg_overhead / (batch_time_ms + avg_overhead)) * 100
    
    return avg_overhead, p95_overhead, impact_pct

if __name__ == "__main__":
    print(f"{'Batch Time':<12} | {'Avg Overhead':<15} | {'P95 Overhead':<15} | {'Throughput Impact':<18}")
    print("-" * 70)
    
    for batch_ms in [500, 200, 50, 5]:
        # Adaptive step count to save time
        steps = 5 if batch_ms >= 500 else 20
        avg, p95, impact = run_benchmark(batch_ms, num_steps=steps)
        print(f"{batch_ms:>7} ms    | {avg:>10.4f} ms    | {p95:>10.4f} ms    | {impact:>15.4f} %")
