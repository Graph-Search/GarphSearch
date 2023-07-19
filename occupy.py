import torch

a = torch.zeros(int(500000000 * 9)).to('cuda:6')

breakpoint()