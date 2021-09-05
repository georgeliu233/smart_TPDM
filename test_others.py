# import numpy as np
# act_set = ["keep_lane","slow_down","change_lane_left","change_lane_right"]

# #print(act_set[np.random.randint(0,4)])
import torch

# a = torch.zeros(1, *(80,80,3))
# print(a.size())
print(torch.cuda.is_available())

dic = {
    '1':[1,2],
    "2":[2,3]
}

buf_dict = dict()
buf_dict['1'] = dic['1']
print(buf_dict)
del dic['1']
print(buf_dict,dic)
