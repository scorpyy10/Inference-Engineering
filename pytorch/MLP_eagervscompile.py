import torch
import time

device="cuda" if torch.cuda.is_available() else "cpu"

print("pytorch:",torch.__version__)
print("Device:",device)

if device =="cuda":
    print("GPU:",torch.cuda.get_device_name())

import torch.nn as nn

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1024,2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,10)

        )
    def forward(self,x):
        return self.network(x)


model=SmallMLP().to(device)
model.eval()

x=torch.randn(32,1024,device=device)

eager_model=model

with torch.inference_mode():
    for _ in range(20):
        output=eager_model(x)

print("Eager output:", output.shape)

compiled_model = torch.compile(model)

with torch.inference_mode():
    for _ in range(20):
        output = compiled_model(x)

print("Compiled output:", output.shape)


def benchmark(model, x, iterations=200):
    times = []

    with torch.inference_mode():
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append((end - start) * 1000)

    return sum(times) / len(times)


# Warmup
with torch.inference_mode():
    for _ in range(30):
        eager_model(x)
        compiled_model(x)

eager_time = benchmark(eager_model, x)
compiled_time = benchmark(compiled_model, x)

speedup = eager_time / compiled_time

print(f"Eager:    {eager_time:.4f} ms")
print(f"Compiled: {compiled_time:.4f} ms")
print(f"Speedup:  {speedup:.2f}x")
print(f"Improvement: {(1 - compiled_time/eager_time)*100:.2f}%")
