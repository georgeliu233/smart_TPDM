from math import ceil

def cnn_calculater(img_input,cnn_params_inputs):
    """
    img_input:(H,W,C)
    cnn_params_inputs:list of cnn layer params:
    [   
        out_channels-(num of filters),
        kernel_size-K:(K*K default ,or (K_h,K_w)),
        strides:S,
        padding:P
    ]
    pooling case: the same
    """
    H,W,C = img_input
    for param in cnn_params_inputs:
        out_channel , kernel_size,stride,pad = param
        if isinstance(kernel_size,list):
            H = (H-kernel_size[0]+2*pad)/stride+1
            W = (W-kernel_size[1]+2*pad)/stride+1
        else:
            H = ceil((H-kernel_size+2*pad)/stride)+1
            W = ceil((W-kernel_size+2*pad)/stride)+1
        C = out_channel
    print(H,W,C)

img_input = (80,80,3)
cnn_params_input = [
    (32,8,4,0),
    (64,4,2,0),
    (64,3,1,0)
]
cnn_calculater(img_input,cnn_params_input)