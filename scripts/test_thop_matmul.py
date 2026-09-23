#!/usr/bin/env python3
import torch
from thop import profile

# Test if thop sees matmul
class SimpleMatmul(torch.nn.Module):
    def forward(self, x):
        return torch.matmul(x, x.transpose(1, 2))

model = SimpleMatmul()
x = torch.randn(1, 98, 48)
macs, _ = profile(model, inputs=(x,), verbose=False)
print(f"Simple matmul: thop={macs:,}, expected={98*98*48:,}")

# Test einsum
class SimpleEinsum(torch.nn.Module):
    def forward(self, q, k):
        return torch.einsum('bhid,bhjd->bhij', q, k)

model2 = SimpleEinsum()
q = torch.randn(1, 8, 98, 6)
k = torch.randn(1, 8, 98, 6)
macs2, _ = profile(model2, inputs=(q, k), verbose=False)
print(f"Einsum: thop={macs2:,}, expected={8*98*98*6:,}")