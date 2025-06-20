import numpy as np
from . import cnodes

def split_shapelist_into_chunks(shapelist,H,W):
    outlist = []
    for shape in shapelist:
        chunk_shape_list = split_shape_into_chunks(shape,H,W)
        outlist.extend(np.array(chunk_shape_list).reshape(-1,2))
    return outlist

def split_shape_into_chunks(shape,H,W):
    '''
    Like split_matrix_into_chunks, but works with just shapes.
    '''
    matrix = np.empty(shape) 
    out = []
    if matrix.shape[1] > W:
        reps = matrix.shape[1]//W + (matrix.shape[1]%W!=0) #ceil func
        for i in range(reps):
            out.append(matrix[:,W*i:W*i+W])
    else:
        out.append(matrix)

    out2 = []
    for i,mat in enumerate(out):
        out_col = []
        if mat.shape[0] > H:
            reps = mat.shape[0]//H + (mat.shape[0]%H!=0) #ceil func
            for i in range(reps):
                out_col.append(mat[H*i:H*i+H])
        else:
            out_col.append(mat)
        out2.append(out_col)

    return [[i.shape for i in c] for c in out2]

def split_matrix_into_chunks(matrix,H,W):
    '''
    Splits matrix into chunks of W,H.

    PARAMETERS
    ----------
    matrix : np.ndarray
        matrix to split
    H : int
        max height of submatrices
    W : int
        max width of submatrices
    '''

    out = []
    if matrix.shape[1] > W:
        reps = matrix.shape[1]//W + (matrix.shape[1]%W!=0) #ceil func
        for i in range(reps):
            out.append(matrix[:,W*i:W*i+W])
    else:
        out.append(matrix)

    out2 = []
    for i,mat in enumerate(out):
        out_col = []
        if mat.shape[0] > H:
            reps = mat.shape[0]//H + (mat.shape[0]%H!=0) #ceil func
            for i in range(reps):
                out_col.append(mat[H*i:H*i+H])
        else:
            out_col.append(mat)
        out2.append(out_col)
    return out2

def split_vector_into_chunks(vector : np.array, W : int):
    '''
    Split vector into chunks of at most W
    '''

    if vector.shape is ():
        # single output channel conv -> single-value bias
        vector = np.expand_dims(vector,-1)

    from itertools import zip_longest
    b = list(zip_longest(*(iter(vector),)*W))
    c = [np.array(i) for i in b]
    d = [i[i != np.array(None)] for i in c]
    return d

def split_kernel_into_channels(kernel:np.ndarray,C:int):
    '''
    Splits a kernel into chunks of at most C channels.
    Assumes KCFxFy
    '''

    subkernels = []

    if kernel.shape[1] > C:
        reps = kernel.shape[1]//C + (kernel.shape[1]%C!=0)
        for i in range(reps):
            subkernels.append(kernel[:,C*i:C*i+C])

    return subkernels

def split_dwc_into_chunks(cnode:cnodes.dwc_node,C:int):
    '''
    Splits a depthwise convolution node into chunks of at most C channels.
    '''
    subkernels = split_kernel_into_channels(cnode.kernel,C)
    strides = cnode.strides

    if len(subkernels) == 1:
        return [cnode]

    nodes = []

    tplitz_output_edge = cnode.outputs[0] + '_tplitz'
    tplitz_node = cnodes.toeplitzizer_node(cnode.inputs,[tplitz_output_edge],ksize=ksize,strides=strides)
    nodes.append(tplitz_node)

    cat_inputs = []

    for i,col in enumerate(submatrices):

        adder_inputs = []

        for j,matrix in enumerate(col):

            # In the crossbar these would be rows, but in the current orientation it's cols
            input_cols = [H*j,H*j+H]

            slicer_output_edge = f'{cnode.outputs[0]}_slicer_{i}-{j}'
            slicer = cnodes.slicer_node([tplitz_output_edge],[slicer_output_edge],col_lim=input_cols)
            
            gemm_output_edge = f'{cnode.outputs[0]}_gemm_{i}-{j}'
            gemm = cnodes.gemm_node([slicer_output_edge],[gemm_output_edge],matrix)

            nodes.append(slicer)
            nodes.append(gemm)

            adder_inputs.append(gemm_output_edge)

        #apply bias to the last matrices
        nodes[-1].biases = subbiases[i]

        if len(adder_inputs) > 1:
            adder_output_edge = f'{cnode.outputs[0]}_adder_{i}'
            adder = cnodes.add_node(adder_inputs,[adder_output_edge])
            nodes.append(adder)
            cat_inputs.append(adder_output_edge)
        else:
            cat_inputs.extend(adder_inputs)

    cat_output_edge = f'{cnode.outputs[0]}_cat'
    cat = cnodes.cat_node(cat_inputs,[cat_output_edge],axis=-1)
    nodes.append(cat)

    C = cnode.matrix.shape[1]

    output_tensorizer = cnodes.reshaper_node([cat_output_edge],cnode.outputs,channels = C)
    nodes.append(output_tensorizer)

    if ksize == 1:
        for node in nodes:
            node.from_type = 'pointwise'
    else:
        for node in nodes:
            node.from_type = 'conv'

    return nodes

def split_conv_into_channels(cnode:cnodes.conv_node,C:int):

    kernels = split_kernel_into_channels(cnode.kernel,C)

    if len(kernels) == 1:
        return [cnode]
    
    nodes = []

    for i,kernel_subC in enumerate(kernels):
        # Create a channel slicing node
        channel_slicer_output = cnode.outputs[0] + f'_ch_{i}'
        conv_output = cnode.outputs[0] + f'_conv_ch_{i}'

        channel_slicer = cnodes.channel_slicing_node(
            inputs           = cnode.inputs,
            outputs          = [channel_slicer_output],
            channel_lim_low  = C*i,                     # i, C, 2C...
            channel_lim_high = C*i + C,                 # can go over the number of channels
            channel_minor    = cnode.channel_minor
        )
        
        # Create a new conv node for each channel set
        new_conv = cnodes.conv_node(
            inputs        = [channel_slicer_output],
            outputs       = [conv_output],
            kernel        = kernel_subC,
            strides       = cnode.strides,
            biases        = cnode.biases[i] if cnode.biases is not None else None,
            channel_minor = cnode.channel_minor,
        )

        nodes.append(channel_slicer)
        nodes.append(new_conv)

    # Create a sum node to combine the outputs of the conv nodes

    return

def split_conv_into_chunks(cnode:cnodes.conv_node,H:int,W:int):
    submatrices = split_matrix_into_chunks(cnode.matrix,H,W)
    subbiases = split_vector_into_chunks(cnode.biases,W)
    ksize = cnode.kernel.shape[-1]
    strides = cnode.strides

    if len(submatrices) == 1 and len(submatrices[0]) == 1:
        if ksize == 1:
            cnode.from_type = 'pointwise' # I don't actually remember what the from_type is for
        else:
            cnode.from_type = 'conv'
        return [cnode]

    nodes = []

    tplitz_output_edge = cnode.outputs[0] + '_tplitz'
    tplitz_node = cnodes.toeplitzizer_node(cnode.inputs,[tplitz_output_edge],ksize=ksize,strides=strides)
    nodes.append(tplitz_node)

    cat_inputs = []

    for i,col in enumerate(submatrices):

        adder_inputs = []

        for j,matrix in enumerate(col):

            # In the crossbar these would be rows, but in the current orientation it's cols
            input_cols = [H*j,H*j+H]

            slicer_output_edge = f'{cnode.outputs[0]}_slicer_{i}-{j}'
            slicer = cnodes.slicer_node([tplitz_output_edge],[slicer_output_edge],col_lim=input_cols)
            
            gemm_output_edge = f'{cnode.outputs[0]}_gemm_{i}-{j}'
            gemm = cnodes.gemm_node([slicer_output_edge],[gemm_output_edge],matrix)

            nodes.append(slicer)
            nodes.append(gemm)

            adder_inputs.append(gemm_output_edge)

        #apply bias to the last matrices
        nodes[-1].biases = subbiases[i]

        if len(adder_inputs) > 1:
            adder_output_edge = f'{cnode.outputs[0]}_adder_{i}'
            adder = cnodes.add_node(adder_inputs,[adder_output_edge])
            nodes.append(adder)
            cat_inputs.append(adder_output_edge)
        else:
            cat_inputs.extend(adder_inputs)

    cat_output_edge = f'{cnode.outputs[0]}_cat'
    cat = cnodes.cat_node(cat_inputs,[cat_output_edge],axis=-1)
    nodes.append(cat)

    C = cnode.matrix.shape[1]

    output_tensorizer = cnodes.reshaper_node([cat_output_edge],cnode.outputs,channels = C)
    nodes.append(output_tensorizer)

    if ksize == 1:
        for node in nodes:
            node.from_type = 'pointwise'
    else:
        for node in nodes:
            node.from_type = 'conv'

    return nodes

def split_gemm_into_chunks(cnode:cnodes.gemm_node,H:int,W:int):

    submatrices = split_matrix_into_chunks(cnode.matrix,H,W)
    subbiases = split_vector_into_chunks(cnode.biases,W)

    if len(submatrices) == 1:
        if len(submatrices[0]) == 1:
            return [cnode]

    nodes = []

    cat_inputs = []

    for i,col in enumerate(submatrices):

        adder_inputs = []

        for j,matrix in enumerate(col):

            # In the crossbar these would be rows, but in the current orientation it's cols
            input_cols = [H*j,H*j+H]

            slicer_output_edge = f'{cnode.outputs[0]}_slicer_{i}-{j}'
            slicer = cnodes.slicer_node(cnode.inputs,[slicer_output_edge],col_lim=input_cols)
            
            gemm_output_edge = f'{cnode.outputs[0]}_gemm_{i}-{j}'
            gemm = cnodes.gemm_node([slicer_output_edge],[gemm_output_edge],matrix)

            nodes.append(slicer)
            nodes.append(gemm)

            adder_inputs.append(gemm_output_edge)

        #apply bias to the last matrices
        nodes[-1].biases = subbiases[i]

        adder_output_edge = f'{cnode.outputs[0]}_adder_{i}'
        adder = cnodes.add_node(adder_inputs,[adder_output_edge])
        nodes.append(adder)

        cat_inputs.append(adder_output_edge)

    cat = cnodes.cat_node(cat_inputs,cnode.outputs,axis=0)
    nodes.append(cat)

    for node in nodes:
        node.from_type = 'gemm'

    return nodes