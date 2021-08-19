import numpy as np
act_set = ["keep_lane","slow_down","change_lane_left","change_lane_right"]

#print(act_set[np.random.randint(0,4)])
import torch

a = torch.zeros(1, *(80,80,3))
print(a.size())
print(torch.cuda.is_available())