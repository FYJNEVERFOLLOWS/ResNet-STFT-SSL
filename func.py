import numpy as np
import torch

def angular_distance(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)

def get_top2_doa(output):
    # print(f'output {output.tolist()}')
    fst = torch.max(output, 0)[1].item()
    temp = torch.roll(output, int(180 - fst), 0)
    temp[180 - 15 : 180 + 15] = 0
    sec = torch.max(torch.roll(temp, int(fst - 180)), 0)[1].item()
    return np.array([fst, sec])