import numpy as np

def get_recfield_for_pixel(r,c,matrix,ksize):
    'Obtains the receptive field for a convolution output pixel'
    if ksize == 1:
        return matrix[r,c]
    if ksize == 3:
        return matrix[r:r+3,c:c+3].transpose(2,0,1)

def toeplitzize_input(in_tensor,ksize=3,strides=1,channel_minor = False, zero_point = 0):
    '''
    Flattens input tensor into a Toeplitz matrix for passing into a
    flattened kernel. Zero pads by default.

    input: B,C,H,W tensor

    Assumes B=1 for now
    '''

    #Convert to B,H,W,C tensor
    tensor = in_tensor.transpose(1,2,0)

    H = tensor.shape[0] // strides
    W = tensor.shape[1] // strides
    C = tensor.shape[2] 

    if ksize == 3:
        tensor2 = np.pad(tensor,((1,1),(1,1),(0,0)), 
                         mode='constant', constant_values=zero_point)
        out = np.empty((H*W,C*9), dtype=tensor.dtype)
    else:
        tensor2 = tensor
        out = np.empty((H*W,C), dtype=tensor.dtype)
    for r in range(H):
        for c in range(W):
            recfield = get_recfield_for_pixel(strides*r,strides*c,tensor2,ksize)
            if channel_minor and ksize == 3:
                out[r*W + c] = recfield.transpose(1,2,0).flatten()
            else:
                out[r*W + c] = recfield.flatten()

    return out
