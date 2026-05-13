from .cnodes import (
    conv_node, gemm_node, add_node, clip_node, flatten_node,
    global_avg_node, quantize_node, from_QLinearConv,
    quantized_linear_add_node, quantized_global_avg_pool_node,
    dequantize_node, shape_node, reshape_node, gather_node,
    unsqueeze_node, squeeze_node, concat_node, from_QLinearMatMul,
    slicer_node,
)


def get_cnode_from_onnx_node(node, nx_model, **kwargs):
    if node.op_type == 'Conv':
        if node.attribute[1].i == 1:
            return conv_node.from_onnx_node(nx_model, node)
        else:
            convs, catter = conv_node.from_onnx_depthwise(nx_model, node)
            return convs + [catter]
    elif node.op_type == 'Gemm':
        return gemm_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Add':
        return add_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Clip':
        return clip_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Flatten':
        return flatten_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'GlobalAveragePool':
        return global_avg_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'QuantizeLinear':
        return quantize_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'QLinearConv':
        return from_QLinearConv(nx_model, node, **kwargs)
    elif node.op_type == 'QLinearAdd':
        return quantized_linear_add_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'QLinearGlobalAveragePool':
        return quantized_global_avg_pool_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'DequantizeLinear':
        return dequantize_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Shape':
        return shape_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Reshape':
        return reshape_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Gather':
        return gather_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Unsqueeze':
        return unsqueeze_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Squeeze':
        return squeeze_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Concat':
        return concat_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'QLinearMatMul':
        return from_QLinearMatMul(nx_model, node)
    elif node.op_type == 'QLinearAveragePool':  # For now, all average pools are global
        return quantized_global_avg_pool_node.from_onnx_node(nx_model, node)
    elif node.op_type == 'Slice':
        return slicer_node.from_onnx_node(nx_model, node)
    else:
        raise NotImplementedError(f'Node type {node.op_type} not implemented')
