import json
import matplotlib.pyplot as plt

def smooth(scalar,weight=0.85):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


with open('/home/haochen/SMARTS_test_TPDM/log_loop_fusioned.json','r',encoding='utf-8') as reader:
    r,t = json.load(reader)

with open('/home/haochen/SMARTS_test_TPDM/log_loop_state.json','r',encoding='utf-8') as reader:
    r2,t2 = json.load(reader)

plt.figure()
plt.plot(t,smooth(r,0.95))
plt.plot(t2,smooth(r2,0.95))
plt.legend(['fusioned','state'])
plt.savefig('log_fusioned_comp.png')