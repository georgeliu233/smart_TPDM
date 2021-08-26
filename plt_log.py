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

def comp():
    with open('/home/haochen/SMARTS_test_TPDM/log_loop_fusioned.json','r',encoding='utf-8') as reader:
        r,t = json.load(reader)

    with open('/home/haochen/SMARTS_test_TPDM/log_loop_state.json','r',encoding='utf-8') as reader:
        r2,t2 = json.load(reader)



    plt.figure()
    plt.plot(t,smooth(r,0.95))
    plt.plot(t2,smooth(r2,0.95))
    plt.legend(['fusioned','state'])
    plt.savefig('log_fusioned_comp.png')

def plot_comp(weight=0.95):
    abs_path = '/home/haochen/SMARTS_test_TPDM/'
    # json_list = [
    #     "log_loop_fusioned",
    #     'log_loop_state',
    #     'log_loop_cnn'
    # ]
    json_list = ['log_left_state']
    data_list = []
    for path in json_list:
        with open(abs_path+path+'.json','r',encoding='utf-8') as reader:
            r,t = json.load(reader)
            data_list.append([r,t])
    
    plt.figure()
    for data in data_list:
        plt.plot(data[1],smooth(data[0],weight))
    #plt.legend(['fusioned','state','CNN'])
    plt.savefig(abs_path+'log_left.png')

if __name__=='__main__':
    plot_comp()