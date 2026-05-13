import pytest
import onnx
import numpy as np
from hwacctools.comp_graph import splitter, cnodes, cgraph, core

def pytest_addoption(parser):
    parser.addoption(
        "--modelpath",
        action="append",
        default=[],
        help="onnx model file path",
    )
    parser.addoption(
        "--core_size",
        action="append",
        default=[(256,256)],
        help="core size for accelerator",
    )

def pytest_generate_tests(metafunc):
    if "modelpath" in metafunc.fixturenames:
        if metafunc.config.getoption("modelpath") == []:
            metafunc.parametrize("modelpath", ['onnx_models/mobilenetv2-12-int8.onnx'])
        else:
            metafunc.parametrize("modelpath", metafunc.config.getoption("modelpath"))
    if "core_size" in metafunc.fixturenames:
        metafunc.parametrize("core_size", metafunc.config.getoption("core_size"))

@pytest.fixture
def nx_model(modelpath):
    nx_model = onnx.load(modelpath)
    return nx_model

@pytest.fixture
def img_array():
    from torchvision import transforms
    from PIL import Image
    img = Image.open('images/imagenet_finch.jpeg')
    img_tensor = transforms.ToTensor()(img).float()
    img_tensor = transforms.CenterCrop(224)(img_tensor)
    # tensor_input = img_tensor.unsqueeze()
    img_array = np.array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@pytest.fixture
def input_dict(nx_model,img_array):
    input_name = nx_model.graph.input[0].name
    input_dict = {input_name: img_array}
    return input_dict

@pytest.fixture
def cgraph_uut(nx_model,img_array):
    cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model)
    # cgraph_UUT.forward(input_dict, recalculate=False) 
    return cgraph_UUT

@pytest.fixture
def core_packed(cgraph_uut,core_size):
    return core.packed_model(cgraph_uut,core_size)
    