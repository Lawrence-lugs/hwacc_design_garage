"""ONNX model manipulation — inference helpers, graph surgery, and initialiser access."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnx.onnx_pb import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
)

def add_tensor_to_model_outputs(model, tensor_name):
    layer_value_info = helper.ValueInfoProto()
    layer_value_info.name = tensor_name
    model.graph.output.append(layer_value_info)
    return model

def get_intermediate_tensor_value(modelpath, tensor_name, input_dict=None):
    """
    Get the value of an intermediate tensor from an ONNX model.
    Works on a copy of the model to preserve the original.
    
    Args:
        modelpath: Path to the ONNX model or the model itself
        tensor_name: Name of the tensor to extract
        input_dict: Dictionary of input tensors {name: value}
    
    Returns:
        The value of the specified tensor
    """
    # Load model and create a copy
    if type(modelpath) == str:
        original_model = onnx.load(modelpath)
    else:
        original_model = modelpath
    
    # Create a deep copy of the model
    model = onnx.ModelProto()
    model.CopyFrom(original_model)
    
    # Create a new model with the specified tensor as output
    model = add_tensor_to_model_outputs(model, tensor_name)

    # Remove inputs from model that are not in input_dict
    input_names = list(input_dict.keys())
    model_inputs = list(model.graph.input)
    for input_info in model_inputs:
        if input_info.name not in input_names:
            model.graph.input.remove(input_info)

    model_outputs = list(model.graph.output)
    for output_info in model_outputs:
        if output_info.name != tensor_name:
            model.graph.output.remove(output_info)

    # Handle intermediate tensors as inputs
    for input_tensor_name,input_tensor in input_dict.items():
        if input_tensor_name not in [inp.name for inp in model.graph.input]:
            # print(f'Input tensor {input_tensor_name} is not original input. Input tensor name provided:', input_tensor_name)
            
            # Rename node outputs that match input_tensor_name
            for node in model.graph.node:
                for i, output_name in enumerate(node.output):
                    if output_name == input_tensor_name:
                        new_output_name = f"{output_name}_original"
                        node.output[i] = new_output_name
                        # print(f"Renamed node output from {input_tensor_name} to {new_output_name}")
            
            # Add the intermediate tensor as a new input with proper type information
            if hasattr(input_tensor, 'dtype'):
                elem_type = onnx.helper.np_dtype_to_tensor_dtype(input_tensor.dtype)
            else:
                elem_type = onnx.TensorProto.UINT8
            
            input_info = helper.make_tensor_value_info(
                name=input_tensor_name,
                elem_type=elem_type,
                shape = None
            )
            model.graph.input.append(input_info)
    
    # Remove nodes that are now unconnected to the main graph
    used_inputs = set([inp.name for inp in model.graph.input])
    used_outputs = set([out.name for out in model.graph.output])
    original_outputs = set([out.name for out in model.graph.output])
    obtained_outputs = set()

    # Process nodes to find all connections
    connected_nodes = []
    changed = True
    while changed:
        changed = False
        
        if original_outputs.issubset(obtained_outputs):
            break

        for node in model.graph.node:

            if original_outputs.issubset(obtained_outputs):
                break

            if node in connected_nodes:
                continue
                
            is_connected = False
            for inp in node.input:
                if inp in used_outputs or inp in used_inputs:
                    is_connected = True
                    obtained_outputs.update(node.output)
                    break
                    
            if is_connected:
                connected_nodes.append(node)
                for out in node.output:
                    used_outputs.add(out)
                changed = True

    # Create a new graph with only the connected nodes
    new_nodes = [node for node in model.graph.node if node in connected_nodes]
    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)

    # Keep only used initializers
    used_initializers = set()
    for node in model.graph.node:
        used_initializers.update(node.input)
    
    new_initializers = [
        init for init in model.graph.initializer 
        if init.name in used_initializers or init.name in used_outputs
    ]
    
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_initializers)

    return infer(model, input_dict)[-1]

def infer(nx_model, input_dict):
    session = ort.InferenceSession(nx_model.SerializeToString())
    outputs = session.run(None, input_dict)
    return outputs

def _extract_value_info(
    input: list[Any] | np.ndarray | None,
    name: str,
    type_proto: TypeProto | None = None,
) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError(
                "_extract_value_info: both input and type_proto arguments cannot be None."
            )
        elif isinstance(input, list):
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(input[0].dtype)
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        elif isinstance(input, TensorProto):
            elem_type = input.data_type
            shape = tuple(input.dims)
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
        else:
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(input.dtype)
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)

    return onnx.helper.make_value_info(name, type_proto)

def expect(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    feeds = {name: value for name, value in zip(node.input, inputs)}
    results = sess.run(None, feeds)
    for expected, output in zip(outputs, results):
        np.allclose(expected, output)

def infer_node_output(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    feeds = {name: value for name, value in zip(node.input, inputs)}
    results = sess.run(None, feeds)
    return results
    

def is_initializer(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return 'initializer'
    return False

def get_initializer_by_name(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return init
    raise LookupError(f'Could not find initializer with name {name}')

def get_node_by_output(onnx_model,output_name):
    for node in onnx_model.graph.node:
        if node.output[0] == output_name:
            return node
    raise LookupError(f'Could not find node with output {output_name}')

def get_attribute_by_name(name:str,attr_list:list):
    for i,attr in enumerate(attr_list):
        if attr.name == name:
            return attr
    raise AttributeError


def delete_initializer_by_name(model, initializer_name):
    for i, init in enumerate(model.graph.initializer):
        if init.name == initializer_name:
            del model.graph.initializer[i]
            break

def randomize_initializer_to_binary(model, initializer_name):
    array = numpy_helper.to_array(get_initializer_by_name(model, initializer_name))
    new_value = np.random.randint(0, 2, size=array.shape).astype(array.dtype)
    tensor = numpy_helper.from_array(new_value, name=initializer_name)
    delete_initializer_by_name(model, initializer_name)
    model.graph.initializer.append(tensor)

def randomize_model_to_binary_weights(model):
    for i,node in enumerate(model.graph.node):
        if node.op_type == 'QLinearConv':
            group = get_attribute_by_name('group', node.attribute).i
            if(group == 1):
                inii = node.input[3]
                randomize_initializer_to_binary(model, inii)
        if node.op_type == 'QLinearMatMul':
            inii = node.input[3]
            randomize_initializer_to_binary(model, inii)
    return model

import onnx
from onnx import helper, numpy_helper

def make_single_node_model(nx_node, initializer_dict, input_names, output_names):
    """
    Create an ONNX ModelProto with a single node and a set of initializers.

    Args:
        nx_node (onnx.NodeProto): The ONNX node to include in the model.
        initializer_dict (dict): Dictionary of {name: np.ndarray} for initializers.

    Returns:
        onnx.ModelProto: The constructed ONNX model.
    """
    # Collect input/output names

    # Create ValueInfoProto for inputs/outputs
    inputs_vi = []
    for name in input_names:
        # If the input is in initializers, get shape/type from the array
        if name in initializer_dict:
            arr = initializer_dict[name]
            vi = helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape)
        else:
            # Unknown shape/type, use float32 [1]
            vi = helper.make_tensor_value_info(name, onnx.TensorProto.UINT8, ['N', 'C', 'H', 'W'])
        inputs_vi.append(vi)

    outputs_vi = [helper.make_tensor_value_info(name, onnx.TensorProto.UINT8, ['N', 'C', 'H', 'W']) for name in output_names]

    # Create initializers
    initializers = [
        numpy_helper.from_array(arr, name=name)
        for name, arr in initializer_dict.items()
    ]

    # Build the graph
    graph = helper.make_graph(
        nodes=[nx_node],
        name="single_node_graph",
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializers,
    )

    # Build the model
    model = helper.make_model(graph, producer_name="make_single_node_model")
    return model