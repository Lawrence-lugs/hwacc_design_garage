"""Quantisation utilities — QuantizedTensor, fixed-point conversion, and bit packing."""

from __future__ import annotations

# Transferred in from https://github.com/Lawrence-lugs/eyeriss
# Ended up useful for a bunch of other work

import numpy as np
import pandas as pd
from scipy.signal import convolve2d

df = pd.DataFrame

class QuantizedTensor:
    def __init__(self,shape=None,precision=None,mode='symmetric',real_values=None,quantized_values=None,scale=None,zero_point=None):
        '''
        Quantized Tensor
        Reals are normalized to [-1,1]
        If mode is 'symmetric', zero point is 0 and scale is 2 / (2**precision - 1)

        Parameters:
        shape : tuple -- Shape of the tensor
        precision : int -- Number of bits to quantize to
        mode : str -- Quantization mode, 'symmetric', 'maxmin' or '3sigma'
        real_values : np.ndarray -- Real values of the tensor
        quantized_values : np.ndarray -- Quantized values of the tensor
        scale : float -- Scale of the quantization
        zero_point : float -- Zero point of the quantization
        
        Main attributes:
        real_values : np.ndarray -- Real values of the tensor
        quantized_values : np.ndarray -- Quantized values of the tensor
        scale : float -- Scale of the quantization
        zero_point : float -- Zero point of the quantization
        shape : tuple -- Shape of the tensor
        fake_quantized_values : np.ndarray -- Quantized values translated back to real range
        '''
        if real_values is not None:
            if precision is None:
                raise ValueError('Precision must be provided')
            self.real_values = real_values
            self.quantize(precision,mode,zero_point=zero_point)
            self.dequantize()
        elif quantized_values is not None:
            if scale is None or zero_point is None:
                raise ValueError('Scale and zero point must be provided')
            self.real_values = None
            self.quantized_values = quantized_values
            
            if type(scale) != np.ndarray:
                self.scale = np.array([scale])
            else:
                self.scale = scale

            self.zero_point = zero_point
            self.dequantize()
        else:
            if shape is None or precision is None:
                raise ValueError('Shape and precision must be provided')
            self.real_values = np.random.uniform(-1,1,shape)
            self.quantize(precision,mode)
            self.dequantize()
        self.shape = self.real_values.shape
        return

    def quantize(self, precision, mode='symmetric', zero_point = None, signed=True):
        if mode == 'maxmin' : 
            clip_high = self.real_values.max()
            clip_low = self.real_values.min()
            # self.scale = 2*max(clip_high,clip_low) / (2**precision - 1)
            self.scale = 2*max(clip_high,clip_low) / (2**precision - 1)
            if zero_point is None:
                if not signed:  
                    self.zero_point = 2 ** (precision - 1) - 1
                self.zero_point = 0
            else:
                self.zero_point = zero_point
        elif mode == '3sigma' :
            mean = self.real_values.mean()
            std = self.real_values.std()
            self.scale = std*3 / (2**precision - 1)
            if zero_point is None:
                # if not signed: 
                self.zero_point = 2 ** (precision - 1) - 1
                # self.zero_point = mean / self.scale
            else:
                self.zero_point = zero_point
        elif mode == 'symmetric':
            self.scale = 2 / (2**precision - 1)
            self.zero_point = 0    

        self.scale = np.array([self.scale]) 

        # Clipping
        if signed:
            clip_high = 2 ** (precision - 1) - 1
            clip_low = -2 ** (precision - 1)
        else:
            clip_high = 2 ** precision - 1
            clip_low = 0

        self.quantized_values = np.round( (self.real_values / self.scale) + self.zero_point ).astype(int)
        # self.quantized_values = np.clip(self.quantized_values, clip_low, clip_high)

    def dequantize(self):
        " Creates fake quantization values from the quantized values, like EdMIPS "
        fp_m = self.scale.reshape(self.scale.shape[0],*([1]*(len(self.quantized_values.shape)-1)))
        self.fake_quantized_values = (self.quantized_values - self.zero_point) * fp_m
        if self.real_values is None:
            self.real_values = self.fake_quantized_values
        return self.fake_quantized_values

    def __repr__(self):
        return self.quantized_values.__repr__()

def convolve_fake_quantized(a,w) -> np.ndarray:
    " o = aw "
    return convolve2d(a.fake_quantized_values,w.fake_quantized_values[::-1].T[::-1].T,mode='valid')

def convolve_reals(a,w) -> np.ndarray:
    " o = aw "
    return convolve2d(a.real_values,w.real_values[::-1].T[::-1].T,mode='valid')

def scaling_quantized_matmul(w,a,outBits,internalPrecision,out_scale = None) -> QuantizedTensor:
    " o = w @ a.T but quantized "

    # Matrix multiplication
    qaqw = w.quantized_values @ a.quantized_values.T

    # Scaling
    if out_scale is None:
        out_scale = 2 / (2**outBits - 1)
    newscale = a.scale * w.scale / out_scale
    m0, shift = convert_scale_to_shift_and_m0(
                    newscale,
                    precision=internalPrecision
                )

    fp_m = m0 * 2**(shift)
    # m0bin = convert_to_fixed_point(m0, internalPrecision)
    # m0int = int(m0bin,base=2)

    scaled_clipped_shifted = (qaqw * fp_m).astype(int)
    o_q = saturating_clip(scaled_clipped_shifted, outBits)

    # Quantized tensor of the output
    o_qtensor = QuantizedTensor(
                    qaqw.shape,
                    outBits,
                    quantized_values=o_q,
                    scale=out_scale,
                    zero_point=0 # Assume 0 for now, might be bad later.
                )

    return o_qtensor

def scaling_quantized_convolution(a,w,outBits,internalPrecision, out_scale = None) -> QuantizedTensor:
    " o = a * w but quantized "

    # Convolution
    qaqw = convolve2d(a.quantized_values,w.quantized_values[::-1].T[::-1].T,mode='valid')

    # Scaling
    if out_scale is None:
        out_scale = 2 / (2**outBits - 1)
    newscale = a.scale * w.scale / out_scale
    m0, shift = convert_scale_to_shift_and_m0(
                    newscale,
                    precision=internalPrecision
                )

    fp_m = m0 * 2**(shift)

    # Reals accounting for quantized and fixed point error
    o_q = (qaqw * fp_m).astype(int)
    o_q = saturating_clip(o_q, outBits)

    # Quantized tensor of the output
    o_qtensor = QuantizedTensor(
                    qaqw.shape,
                    outBits,
                    quantized_values=o_q,
                    scale=out_scale,
                    zero_point=0 # Assume 0 for now, might be bad later.
                )

    return o_qtensor
    
def convert_scale_to_shift_and_m0(scale,precision=16):
    " Convert scale(s) to shift and zero point "

    if scale == 0:
        return 0, 0

    shift = int(np.ceil(np.log2(scale)))
    # shift = np.abs(shift)
    m0 = scale / 2**shift

    fp_string = convert_to_fixed_point(m0,precision)
    m0_clipped = fixed_point_to_float(fp_string,precision)
    return m0_clipped, shift

vconvert_scale_to_shift_and_m0 = np.vectorize(convert_scale_to_shift_and_m0)

def convert_to_fixed_point(number,precision):
    " Convert a float (0,1) to fixed point binary string"
    if number == 1:
        number = 0.9999999999999999
    x = number*(2**precision)
    x = np.floor(x).astype(int)
    return np.binary_repr(x,width=precision)

def convert_to_fixed_point_int(number,precision):
    " Convert a float [0,1] to fixed point binary "
    x = number*(2**precision)
    x = np.floor(x).astype(int)
    return x

vconvert_to_fixed_point_int = np.vectorize(convert_to_fixed_point_int)

def fixed_point_to_float(number,precision):
    " Convert a fixed point binary string to float [0,1] "
    # out = 0
    # for i in range(precision):
    #     out += int(number[i]) * 2**-(i+1)
    if len(number) != precision:
        raise ValueError('Number must be of length {}'.format(precision))
    return int(number,2)*(2.**-(precision))

def _right_shift(number,shift):
    " Right shift a number. Shifting rounds toward -infty"
    return int(np.floor(number / 2.**(shift)))

right_shift = np.vectorize(_right_shift)

def get_array_bits(array,signed=True):
    " Get the number of bits required to represent an array "
    return int(np.ceil(np.log2(np.abs(array).max())))+signed

def _saturating_clip (num_i, inBits = 16, outBits = 8, signed = True):
    '''
    Saturating clip.

    This only implemented saturation before. Now it also clips.
    '''

    # floor to round towards negative infinity
    num_i_shifted = right_shift(num_i, inBits - outBits)

    if signed:
        min = -(2**(outBits-1))
        max = 2**(outBits-1)-1
    else:
        min = 0
        max = 2**outBits-1
    
    if(num_i_shifted < min):
        return min
    if(num_i_shifted > max):
        return max
    
    return num_i_shifted

saturating_clip = np.vectorize(_saturating_clip)

def binary_array_to_int(array,signed=False,outBits=None):
    ''' 
    Convert a binary array to an integer 
    The binary array is assumed to be in the last dimension
    [
    [ 0, 1, 1, 0],
    [ 1, 0, 0, 1],
    ...
    ]
    '''
    out = 0
    arr = np.moveaxis(array,-1,0)
    for bit in arr:
        out = (out << 1) | bit
    return out

def array_bin_to_int(array,signed=False,outBits=None):
    shape = array.shape
    pows = np.array([2**i for i in range(shape[-1])])[::-1]
    return np.sum(pows*array,axis=1)

def int_to_bin(n, bits):
    " Convert an integer to a binary array "
    return np.moveaxis(np.array([n >> i & 1 for i in range(bits-1,-1,-1)]),0,-1)

def int_to_trit(n, trits):
    " Convert an integer to a trit (ternary) array "
    a = int_to_bin(abs(n), trits)
    tritter = (((n < 0) * -2) + 1) # Bipolar array
    tritter = tritter.repeat(trits, axis=n.ndim-1).reshape(a.shape)
    return (a * tritter)

def simulate_trit_mul(w,x, trits, verbose=False, real=True):
    " trit-decomposed multiplication "
    x_trits = int_to_trit(x, trits)
    partials = x_trits.T @ w.T

    if(verbose):
        print('x trits:')
        print(df(x_trits))
        print('partials:')
        print(df(partials.T[::-1].T))

    if real:
        return po2_accumulate_real(partials, verbose=verbose)
    else:
        return po2_accumulate(partials, verbose=verbose)

def po2_accumulate(a,verbose=False):
    " Accumulate an array of powers of 2 "
    acc = np.zeros(a.shape[1])
    print('shifting accumulator:')
    for shift,row in enumerate(a[::-1]):
        acc = acc + row * 2**shift
        if verbose: print(acc)
    return acc

def po2_accumulate_real(a,verbose=False):
    " Accumulate an array of powers of 2 "
    acc = np.zeros(a.shape[1])
    print('shifting accumulator:')
    b = a * 2**(a.shape[0]-1)
    accs = []
    for shift,row in enumerate(b[::-1]):
        acc = acc/2
        acc = acc + row
        accs.append(acc.T[::-1].T.astype(int))
    if verbose: print(df(accs))
    return acc

def fprint_as_hex(
    filepath,
    array,
    elementBits = 8,
    wordSize = 32
    ):

    a = array.flatten()

    with open(filepath, 'w') as f:
        for i in range(0, len(a), wordSize // elementBits):
            hex_str = ''.join([f'{int(x):02x}' for x in a[i:i + wordSize // elementBits]])
            f.write(hex_str + '\n')
        f.write('\n')

    return

def as_packed_hex(
    array,
    elementBits = 8,
    wordSize = 32
    ):
    """
    Convert an array to a packed hex string.
    """
    a = array.flatten()
    out = np.array([], dtype=str)

    for i in range(0, len(a), wordSize // elementBits):
        hex_str = ''.join([f'{int(x):02x}' for x in a[i:i + wordSize // elementBits]])
        out = np.append(out, hex_str)

    return out