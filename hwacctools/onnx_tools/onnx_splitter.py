"""ONNX graph splitters — split large conv/matmul nodes to fit AIMC core dimensions."""

from __future__ import annotations

import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

def _extract_attr_dict(node):
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

def _set_group_for_depthwise(attr_dict, new_in_channels, is_depthwise):
    if is_depthwise:
        attr_dict = dict(attr_dict)  # copy
        attr_dict['group'] = new_in_channels
    return attr_dict

def split_qlinearconv_node_per_channel(graph, node, C_max, prefix="split"):
    attr_dict = _extract_attr_dict(node)
    group = attr_dict.get('group', 1)
    weight_name = node.input[3]
    weight_scale_name = node.input[4]
    weight_zp_name = node.input[5]
    weight_init = next(i for i in graph.initializer if i.name == weight_name)
    weight_scale_init = next(i for i in graph.initializer if i.name == weight_scale_name)
    weight_zp_init = next(i for i in graph.initializer if i.name == weight_zp_name)
    W = numpy_helper.to_array(weight_init)
    W_scale = numpy_helper.to_array(weight_scale_init)
    W_zp = numpy_helper.to_array(weight_zp_init)
    C_in = W.shape[1]
    is_depthwise = (group != 1 and C_in == 1)
    new_nodes = []
    new_inits = []
    out_names = []

    for i, c_start in enumerate(range(0, C_in, C_max)):
        c_end = min(c_start + C_max, C_in)
        W_split = W[:, c_start:c_end, :, :]
        w_split_name = f"{prefix}_w_{i}"
        w_split_init = numpy_helper.from_array(W_split, name=w_split_name)
        new_inits.append(w_split_init)

        slice_out_name = f"{prefix}_slice_{i}"
        starts_name = f"{prefix}_starts_{i}"
        ends_name = f"{prefix}_ends_{i}"
        axes_name = f"{prefix}_axes_{i}"
        steps_name = f"{prefix}_steps_{i}"

        starts_init = numpy_helper.from_array(np.array([c_start], dtype=np.int64), name=starts_name)
        ends_init = numpy_helper.from_array(np.array([c_end], dtype=np.int64), name=ends_name)
        axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
        steps_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name)
        new_inits.extend([starts_init, ends_init, axes_init, steps_init])

        slice_node = helper.make_node(
            "Slice",
            inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
            outputs=[slice_out_name],
            name=f"{prefix}_slice_{i}"
        )
        new_nodes.append(slice_node)

        inputs = list(node.input)
        inputs[0] = slice_out_name
        inputs[3] = w_split_name
        out_name = f"{prefix}_out_{i}"
        out_names.append(out_name)
        this_attr_dict = attr_dict # No need to change group for per-C split
        qconv_node = helper.make_node(
            "QLinearConv",
            inputs=inputs,
            outputs=[out_name],
            name=f"{prefix}_qconv_{i}",
            **this_attr_dict
        )
        new_nodes.append(qconv_node)

    sum_out = out_names[0]
    for i in range(1, len(out_names)):
        add_out = f"{prefix}_add_{i}"
        qadd_node = helper.make_node(
            "QLinearAdd",
            inputs=[
                sum_out, 
                node.input[6], node.input[7],
                out_names[i],
                node.input[6], node.input[7],
                node.input[6], node.input[7],
            ],
            outputs=[add_out],
            name=f"{prefix}_qadd_{i}",
            domain="com.microsoft"
        )
        new_nodes.append(qadd_node)
        sum_out = add_out

    return new_nodes, new_inits, sum_out

def replace_qlinearconv_with_split(graph, node, new_nodes, new_inits, final_output):
    node_idx = None
    for idx, n in enumerate(graph.node):
        if n == node:
            node_idx = idx
            break
    if node_idx is not None:
        del graph.node[node_idx]

    for init in new_inits:
        graph.initializer.append(init)
    for n in new_nodes:
        graph.node.append(n)

    orig_output = node.output[0]
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == orig_output:
                n.input[i] = final_output
    for out in graph.output:
        if out.name == orig_output:
            out.name = final_output

def split_qlinearconv_node_per_output_channel(graph, node, K_max, prefix="split_out"):
    """
    Splits a QLinearConv node into multiple QLinearConv nodes with at most K_max output channels,
    then concatenates their outputs to emulate the original node.
    Handles depthwise (group==C) by setting group accordingly and slicing input/output if needed.
    Also splits biases if present.
    """
    attr_dict = _extract_attr_dict(node)
    group = attr_dict.get('group', 1)
    weight_name = node.input[3]
    weight_scale_name = node.input[4]
    weight_zp_name = node.input[5]
    weight_init = next(i for i in graph.initializer if i.name == weight_name)
    weight_scale_init = next(i for i in graph.initializer if i.name == weight_scale_name)
    weight_zp_init = next(i for i in graph.initializer if i.name == weight_zp_name)
    W = numpy_helper.to_array(weight_init)
    W_scale = numpy_helper.to_array(weight_scale_init)
    W_zp = numpy_helper.to_array(weight_zp_init)
    K = W.shape[0]
    C_in = W.shape[1]
    is_depthwise = (group != 1 and C_in == 1)

    # Bias splitting
    has_bias = len(node.input) > 8 and node.input[8] != ""
    if has_bias:
        bias_name = node.input[8]
        bias_init = next(i for i in graph.initializer if i.name == bias_name)
        bias = numpy_helper.to_array(bias_init)

    new_nodes = []
    new_inits = []
    out_names = []

    for i, k_start in enumerate(range(0, K, K_max)):
        k_end = min(k_start + K_max, K)
        W_split = W[k_start:k_end, :, :, :]
        w_split_name = f"{prefix}_w_{i}"
        w_split_init = numpy_helper.from_array(W_split, name=w_split_name)
        new_inits.append(w_split_init)

        if W_scale.ndim == 1 and len(W_scale) == K:
            W_scale_split = W_scale[k_start:k_end]
            w_scale_split_name = f"{prefix}_w_scale_{i}"
            w_scale_split_init = numpy_helper.from_array(W_scale_split, name=w_scale_split_name)
            new_inits.append(w_scale_split_init)
        else:
            w_scale_split_name = weight_scale_name

        if W_zp.ndim == 1 and len(W_zp) == K:
            W_zp_split = W_zp[k_start:k_end]
            w_zp_split_name = f"{prefix}_w_zp_{i}"
            w_zp_split_init = numpy_helper.from_array(W_zp_split, name=w_zp_split_name)
            new_inits.append(w_zp_split_init)
        else:
            w_zp_split_name = weight_zp_name

        # Bias split for this output chunk
        if has_bias:
            bias_split = bias[k_start:k_end] if bias.ndim == 1 and len(bias) == K else bias
            bias_split_name = f"{prefix}_bias_{i}"
            bias_split_init = numpy_helper.from_array(bias_split, name=bias_split_name)
            new_inits.append(bias_split_init)

        inputs = list(node.input)
        inputs[3] = w_split_name
        inputs[4] = w_scale_split_name
        inputs[5] = w_zp_split_name
        if has_bias:
            inputs[8] = bias_split_name

        # For depthwise, slice the input tensor as well (axis=1, same as output channels)
        if is_depthwise:
            slice_out_name = f"{prefix}_in_slice_{i}"
            starts_name = f"{prefix}_in_starts_{i}"
            ends_name = f"{prefix}_in_ends_{i}"
            axes_name = f"{prefix}_in_axes_{i}"
            steps_name = f"{prefix}_in_steps_{i}"

            starts_init = numpy_helper.from_array(np.array([k_start], dtype=np.int64), name=starts_name)
            ends_init = numpy_helper.from_array(np.array([k_end], dtype=np.int64), name=ends_name)
            axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
            steps_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name)
            new_inits.extend([starts_init, ends_init, axes_init, steps_init])

            slice_node = helper.make_node(
                "Slice",
                inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                outputs=[slice_out_name],
                name=f"{prefix}_in_slice_{i}"
            )
            new_nodes.append(slice_node)
            inputs[0] = slice_out_name

        out_name = f"{prefix}_out_{i}"
        out_names.append(out_name)
        this_attr_dict = _set_group_for_depthwise(attr_dict, k_end - k_start, is_depthwise)
        qconv_node = helper.make_node(
            "QLinearConv",
            inputs=inputs,
            outputs=[out_name],
            name=f"{prefix}_qconv_{i}",
            **this_attr_dict
        )
        new_nodes.append(qconv_node)

    concat_out = f"{prefix}_concat_out"
    concat_node = helper.make_node(
        "Concat",
        inputs=out_names,
        outputs=[concat_out],
        name=f"{prefix}_concat",
        axis=1
    )
    new_nodes.append(concat_node)

    return new_nodes, new_inits, concat_out

def split_qlinearconv_node_per_output_and_input_channel(graph, node, K_max, C_max, prefix="split_both"):
    attr_dict = _extract_attr_dict(node)
    group = attr_dict.get('group', 1)
    weight_name = node.input[3]
    weight_init = next(i for i in graph.initializer if i.name == weight_name)
    W = numpy_helper.to_array(weight_init)
    K = W.shape[0]
    C_in = W.shape[1]
    is_depthwise = (group != 1 and C_in == 1)

    # Bias splitting
    has_bias = len(node.input) > 8 and node.input[8] != ""
    if has_bias:
        bias_name = node.input[8]
        bias_init = next(i for i in graph.initializer if i.name == bias_name)
        bias = numpy_helper.to_array(bias_init)

    if W.shape[0] <= K_max and W.shape[1] <= C_max:
        return [node], [], node.output[0]
    if K <= K_max:
        return split_qlinearconv_node_per_channel(graph, node, C_max, prefix)
    if W.shape[1] <= C_max:
        return split_qlinearconv_node_per_output_channel(graph, node, K_max, prefix)

    all_nodes = []
    all_inits = []
    concat_outputs = []

    weight_scale_name = node.input[4]
    weight_zp_name = node.input[5]
    weight_scale_init = next(i for i in graph.initializer if i.name == weight_scale_name)
    weight_zp_init = next(i for i in graph.initializer if i.name == weight_zp_name)
    W_scale = numpy_helper.to_array(weight_scale_init)
    W_zp = numpy_helper.to_array(weight_zp_init)

    for k_idx, k_start in enumerate(range(0, K, K_max)):
        k_end = min(k_start + K_max, K)
        W_k = W[k_start:k_end, :, :, :]
        w_k_name = f"{prefix}_w_{k_idx}"
        w_k_init = numpy_helper.from_array(W_k, name=w_k_name)
        all_inits.append(w_k_init)

        if W_scale.ndim == 1 and len(W_scale) == K:
            W_scale_k = W_scale[k_start:k_end]
            w_scale_k_name = f"{prefix}_w_scale_{k_idx}"
            w_scale_k_init = numpy_helper.from_array(W_scale_k, name=w_scale_k_name)
            all_inits.append(w_scale_k_init)
        else:
            w_scale_k_name = weight_scale_name

        if W_zp.ndim == 1 and len(W_zp) == K:
            W_zp_k = W_zp[k_start:k_end]
            w_zp_k_name = f"{prefix}_w_zp_{k_idx}"
            w_zp_k_init = numpy_helper.from_array(W_zp_k, name=w_zp_k_name)
            all_inits.append(w_zp_k_init)
        else:
            w_zp_k_name = weight_zp_name

        # Split biases as well
        if has_bias:
            bias_split = bias[k_start:k_end]
            bias_split_name = f"{prefix}_bias_{k_idx}"
            bias_split_init = numpy_helper.from_array(bias_split, name=bias_split_name)
            all_inits.append(bias_split_init)

        temp_inputs = list(node.input)
        temp_inputs[3] = w_k_name
        temp_inputs[4] = w_scale_k_name
        temp_inputs[5] = w_zp_k_name
        if has_bias:
            temp_inputs[8] = bias_split_name

        # For depthwise, slice the input tensor as well (axis=1, same as output channels)
        if is_depthwise:
            slice_out_name = f"{prefix}_in_slice_{k_idx}"
            starts_name = f"{prefix}_in_starts_{k_idx}"
            ends_name = f"{prefix}_in_ends_{k_idx}"
            axes_name = f"{prefix}_in_axes_{k_idx}"
            steps_name = f"{prefix}_in_steps_{k_idx}"

            starts_init = numpy_helper.from_array(np.array([k_start], dtype=np.int64), name=starts_name)
            ends_init = numpy_helper.from_array(np.array([k_end], dtype=np.int64), name=ends_name)
            axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
            steps_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name)
            all_inits.extend([starts_init, ends_init, axes_init, steps_init])

            slice_node = helper.make_node(
                "Slice",
                inputs=[temp_inputs[0], starts_name, ends_name, axes_name, steps_name],
                outputs=[slice_out_name],
                name=f"{prefix}_in_slice_{k_idx}"
            )
            all_nodes.append(slice_node)
            temp_inputs[0] = slice_out_name

        W_k_C = W_k
        C_in_k = W_k_C.shape[1]
        out_names = []
        nodes = []
        inits = []

        for c_idx, c_start in enumerate(range(0, C_in_k, C_max)):
            c_end = min(c_start + C_max, C_in_k)
            W_split = W_k_C[:, c_start:c_end, :, :]
            w_split_name = f"{prefix}_w_{k_idx}_{c_idx}"
            w_split_init = numpy_helper.from_array(W_split, name=w_split_name)
            inits.append(w_split_init)

            slice_out_name = f"{prefix}_slice_{k_idx}_{c_idx}"
            starts_name = f"{prefix}_starts_{k_idx}_{c_idx}"
            ends_name = f"{prefix}_ends_{k_idx}_{c_idx}"
            axes_name = f"{prefix}_axes_{k_idx}_{c_idx}"
            steps_name = f"{prefix}_steps_{k_idx}_{c_idx}"

            starts_init = numpy_helper.from_array(np.array([c_start], dtype=np.int64), name=starts_name)
            ends_init = numpy_helper.from_array(np.array([c_end], dtype=np.int64), name=ends_name)
            axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
            steps_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name)
            inits.extend([starts_init, ends_init, axes_init, steps_init])

            slice_node = helper.make_node(
                "Slice",
                inputs=[temp_inputs[0], starts_name, ends_name, axes_name, steps_name],
                outputs=[slice_out_name],
                name=f"{prefix}_slice_{k_idx}_{c_idx}"
            )
            nodes.append(slice_node)

            out_name = f"{prefix}_out_{k_idx}_{c_idx}"
            out_names.append(out_name)
            qconv_inputs = list(temp_inputs)
            qconv_inputs[0] = slice_out_name
            qconv_inputs[3] = w_split_name
            this_attr_dict = _set_group_for_depthwise(attr_dict, k_end - k_start, is_depthwise)
            qconv_node = helper.make_node(
                "QLinearConv",
                inputs=qconv_inputs,
                outputs=[out_name],
                name=f"{prefix}_qconv_{k_idx}_{c_idx}",
                **this_attr_dict
            )
            nodes.append(qconv_node)

        sum_out = out_names[0]
        for i in range(1, len(out_names)):
            add_out = f"{prefix}_add_{k_idx}_{i}"
            qadd_node = helper.make_node(
                "QLinearAdd",
                inputs=[
                    sum_out, 
                    node.input[6], node.input[7],
                    out_names[i],
                    node.input[6], node.input[7],
                    node.input[6], node.input[7],
                ],
                outputs=[add_out],
                name=f"{prefix}_qadd_{k_idx}_{i}",
                domain="com.microsoft"
            )
            nodes.append(qadd_node)
            sum_out = add_out

        all_nodes.extend(nodes)
        all_inits.extend(inits)
        concat_outputs.append(sum_out)

    concat_out = f"{prefix}_concat_out"
    concat_node = helper.make_node(
        "Concat",
        inputs=concat_outputs,
        outputs=[concat_out],
        name=f"{prefix}_concat",
        axis=1
    )
    all_nodes.append(concat_node)

    return all_nodes, all_inits, concat_out

def split_qlinearmatmul_node(graph, node, K_max=None, C_max=None, prefix="split_matmul"):
    """
    Splits a QLinearMatMul node along the K (shared) and/or C (input) dimensions.
    - K_max: max chunk size for the shared dimension (columns of A, rows of B)
    - C_max: max chunk size for the input dimension (columns of B, output dimension)
    """
    # QLinearMatMul: [A, A_scale, A_zero, B, B_scale, B_zero, Y_scale, Y_zero]
    A_name = node.input[0]
    B_name = node.input[3]
    B_scale_name = node.input[4]
    B_zp_name = node.input[5]

    B_init = next(i for i in graph.initializer if i.name == B_name)
    B_scale_init = next(i for i in graph.initializer if i.name == B_scale_name)
    B_zp_init = next(i for i in graph.initializer if i.name == B_zp_name)

    B = numpy_helper.to_array(B_init)
    B_scale = numpy_helper.to_array(B_scale_init)
    B_zp = numpy_helper.to_array(B_zp_init)

    # Assume B shape: (K, N) or (K,) for 1D
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    K_dim, N_dim = B.shape

    new_nodes = []
    new_inits = []
    out_names = []

    # Only split if K_max or C_max is set and less than the dimension
    K_chunks = [(0, K_dim)] if not K_max or K_max >= K_dim else [
        (k, min(k + K_max, K_dim)) for k in range(0, K_dim, K_max)
    ]
    C_chunks = [(0, N_dim)] if not C_max or C_max >= N_dim else [
        (c, min(c + C_max, N_dim)) for c in range(0, N_dim, C_max)
    ]

    for c_idx, (c_start, c_end) in enumerate(C_chunks):
        # Slice B_scale and B_zp if they are per-column (N-dim)
        if B_scale.ndim == 1 and len(B_scale) == N_dim:
            B_scale_split = B_scale[c_start:c_end]
            b_scale_split_name = f"{prefix}_b_scale_{c_idx}"
            b_scale_split_init = numpy_helper.from_array(B_scale_split, name=b_scale_split_name)
            new_inits.append(b_scale_split_init)
        else:
            b_scale_split_name = B_scale_name

        if B_zp.ndim == 1 and len(B_zp) == N_dim:
            B_zp_split = B_zp[c_start:c_end]
            b_zp_split_name = f"{prefix}_b_zp_{c_idx}"
            b_zp_split_init = numpy_helper.from_array(B_zp_split, name=b_zp_split_name)
            new_inits.append(b_zp_split_init)
        else:
            b_zp_split_name = B_zp_name

        out_names_k = []
        for k_idx, (k_start, k_end) in enumerate(K_chunks):
            # Slice B
            B_split = B[k_start:k_end, c_start:c_end]
            b_split_name = f"{prefix}_b_{c_idx}_{k_idx}"
            b_split_init = numpy_helper.from_array(B_split, name=b_split_name)
            new_inits.append(b_split_init)

            # Optionally, slice A if splitting K (columns)
            if K_max and K_max < K_dim:
                # Need to slice A's columns: shape (..., K)
                slice_a_name = f"{prefix}_a_{c_idx}_{k_idx}"
                starts_name = f"{prefix}_a_starts_{c_idx}_{k_idx}"
                ends_name = f"{prefix}_a_ends_{c_idx}_{k_idx}"
                axes_name = f"{prefix}_a_axes_{c_idx}_{k_idx}"
                steps_name = f"{prefix}_a_steps_{c_idx}_{k_idx}"
                starts_init = numpy_helper.from_array(np.array([k_start], dtype=np.int64), name=starts_name)
                ends_init = numpy_helper.from_array(np.array([k_end], dtype=np.int64), name=ends_name)
                axes_init = numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name)
                steps_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name)
                new_inits.extend([starts_init, ends_init, axes_init, steps_init])
                slice_node = helper.make_node(
                    "Slice",
                    inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                    outputs=[slice_a_name],
                    name=f"{prefix}_a_slice_{c_idx}_{k_idx}"
                )
                new_nodes.append(slice_node)
                a_input = slice_a_name
            else:
                a_input = node.input[0]

            # Build QLinearMatMul node
            inputs = list(node.input)
            inputs[0] = a_input
            inputs[3] = b_split_name
            inputs[4] = b_scale_split_name
            inputs[5] = b_zp_split_name
            out_name = f"{prefix}_out_{c_idx}_{k_idx}"
            out_names_k.append(out_name)
            matmul_node = helper.make_node(
                "QLinearMatMul",
                inputs=inputs,
                outputs=[out_name],
                name=f"{prefix}_qmatmul_{c_idx}_{k_idx}"
            )
            new_nodes.append(matmul_node)

        # If K was split, sum the outputs for this C chunk
        sum_out = out_names_k[0]
        if len(out_names_k) > 1:
            for i in range(1, len(out_names_k)):
                add_out = f"{prefix}_add_{c_idx}_{i}"
                qadd_node = helper.make_node(
                    "QLinearAdd",
                    inputs=[
                        sum_out, node.input[6], node.input[7],
                        out_names_k[i], node.input[6], node.input[7],
                        node.input[6], node.input[7],
                    ],
                    outputs=[add_out],
                    name=f"{prefix}_qadd_{c_idx}_{i}",
                    domain="com.microsoft"
                )
                new_nodes.append(qadd_node)
                sum_out = add_out
        out_names.append(sum_out)

    # If C was split, concatenate the outputs along the last axis
    if len(C_chunks) > 1:
        concat_out = f"{prefix}_concat_out"
        concat_node = helper.make_node(
            "Concat",
            inputs=out_names,
            outputs=[concat_out],
            name=f"{prefix}_concat",
            axis=-1
        )
        new_nodes.append(concat_node)
        final_out = concat_out
    else:
        final_out = out_names[0]

    return new_nodes, new_inits, final_out

def split_model_to_per_channel(graph, C_max=256, K_max=256, dwC_max=32, prefix="split", enforce_unroll=True):
    """
    Applies the QLinearConv splitters to every QLinearConv node in the ONNX graph,
    limiting each to at most K_max output channels (dwC_max for depthwise) and C_max input channels.
    Only splits nodes with group=1 or depthwise (group==C).
    Can make sure unrolling the filter does not exceed C_max, for certain hardware constraints.

    THIS FUNCTION MODIFIES THE GRAPH IN PLACE.
    """
    qconv_nodes = [node for node in graph.node if node.op_type == "QLinearConv"]

    for idx, node in enumerate(qconv_nodes):
        # Check if node is depthwise
        attr_dict = _extract_attr_dict(node)
        group = attr_dict.get('group', 1)
        weight_name = node.input[3]
        weight_init = next(i for i in graph.initializer if i.name == weight_name)
        W = numpy_helper.to_array(weight_init)
        C_in = W.shape[1]
        is_depthwise = (group != 1 and C_in == 1)

        # Ensure unrolling the filter doesn't exceed C_max
        kernel_shape = attr_dict.get('kernel_shape', [])
        c_max = C_max // np.prod(kernel_shape) if enforce_unroll else C_max

        # Use dwC_max for depthwise convolutions, K_max otherwise
        k_max = dwC_max if is_depthwise else K_max

        new_nodes, new_inits, final_output = split_qlinearconv_node_per_output_and_input_channel(
            graph, node, k_max, c_max, prefix=f"{prefix}_{idx}"
        )

        if len(new_nodes) == 0:
            continue

        replace_qlinearconv_with_split(graph, node, new_nodes, new_inits, final_output)

    # Handle QLinearMatMul nodes similarly
    qmatmul_nodes = [node for node in graph.node if node.op_type == "QLinearMatMul"]    

    for idx, node in enumerate(qmatmul_nodes):
    
        new_nodes, new_inits, final_output = split_qlinearmatmul_node(
            graph, node, K_max=K_max, C_max=C_max, prefix=f"{prefix}_matmul_{idx}"
        )

        if len(new_nodes) == 0:
            continue

        replace_qlinearconv_with_split(graph, node, new_nodes, new_inits, final_output)

    # Finally, sort the graph nodes topologically to ensure execution order
    topological_sort_onnx_graph(graph)

    return

def topological_sort_onnx_graph(graph):
    """
    Returns a list of nodes in topological (execution) order for the given ONNX graph.\
    
    THIS FUNCTION MODIFIES THE GRAPH IN PLACE.
    """
    from collections import defaultdict, deque

    # Build dependency graph
    input_to_nodes = defaultdict(list)
    node_inputs = {}
    all_inputs = set()
    for node in graph.node:
        node_inputs[node.name] = set(node.input)
        for inp in node.input:
            input_to_nodes[inp].append(node)
            all_inputs.add(inp)

    # Find all available tensors (graph inputs and initializers)
    available = set(inp.name for inp in graph.input)
    available.update(init.name for init in graph.initializer)

    # Nodes with all inputs available
    ready = deque()
    in_degree = {}
    for node in graph.node:
        missing = [inp for inp in node.input if inp not in available]
        in_degree[node.name] = len(missing)
        if in_degree[node.name] == 0:
            ready.append(node)

    sorted_nodes = []
    visited = set()

    while ready:
        node = ready.popleft()
        sorted_nodes.append(node)
        visited.add(node.name)
        for out in node.output:
            for consumer in input_to_nodes.get(out, []):
                if consumer.name in visited:
                    continue
                in_degree[consumer.name] -= 1
                if in_degree[consumer.name] == 0:
                    ready.append(consumer)

    if len(sorted_nodes) != len(graph.node):
        raise RuntimeError("Graph has cycles or disconnected nodes!")

    graph.ClearField('node')
    graph.node.extend(sorted_nodes)

    return graph