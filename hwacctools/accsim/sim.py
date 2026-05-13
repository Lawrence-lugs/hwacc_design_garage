import argparse
import numpy as np
from hwacctools.comp_graph import cgraph, core
import onnx
from torchvision import transforms
from PIL import Image
import hwacctools.onnx_utils as onnx_utils
import pandas as pd

df = pd.DataFrame


def generate_input_array(img_path):
    img = Image.open(img_path)
    img_tensor = transforms.ToTensor()(img).float()
    img_tensor = transforms.CenterCrop(224)(img_tensor)
    img_array = np.array(img_tensor)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_simulation(modelpath, core_size=(256, 256)):
    nx_model = onnx.load(modelpath)
    cgraph_UUT = cgraph.Cgraph.from_onnx_model(nx_model)
    u_packed = core.packed_model(cgraph_UUT, core_size=core_size)
    return u_packed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run accelerator simulation on an ONNX model.")
    parser.add_argument("modelpath", help="Path to the ONNX model file")
    parser.add_argument(
        "--core-size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("H", "W"),
        help="Accelerator core size (default: 256 256)",
    )
    args = parser.parse_args()
    run_simulation(args.modelpath, core_size=tuple(args.core_size))

