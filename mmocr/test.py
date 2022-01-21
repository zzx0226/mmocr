import torch
from chamfer_distance import ChamferDistance
import time

chamfer_dist = ChamferDistance()

p1 = torch.rand([80000, 25, 3])
p2 = torch.rand([80000, 15, 3])

s = time.time()
dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

torch.cuda.synchronize()
print(f"Time: {time.time() - s} seconds")
print(f"Loss: {loss}")