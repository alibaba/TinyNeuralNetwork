import os
import sys

import torch

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter.utils.tensorrt import compare_onnx_tensorrt

# Configurable parameters
# Batch size
USE_FP16 = True
USE_INT8 = False
# Workspace size for tuning TRT models (GB)
WORKSPACE_SIZE = 4

# Dict for keep dynamic shapes when building
# Format: {'input_name': (min_shape, opt_shape, max_shape)}
# Example: {'input': ((1, 3, 32, 32), (1, 3, 64, 64), (1, 3, 224, 224))}
# Suggestion: Make dynamic axes in those shapes as small as possible to avoid OOM
DYNAMIC_SHAPE_BUILD_DICT = {}

# Dict for keep dynamic shapes during evaluation
# Format: {'input_name': shape}
# Example: {'input': (1, 3, 32, 32)}
# If `INPUT_PATH_MAPPING` is specified, then it should also be set
DYNAMIC_SHAPE_EVAL_DICT = {}

# Dict for test data buffers
# Format: {'input_name': 'input_data_path'}
# Example: {'input': 'test.data'}
INPUT_PATH_MAPPING = {}


def main():
    # [IMPORTANT] You need to setup your enviroment with the following dependencies.
    # tensorrt https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing
    # onnxruntime-gpu
    # pycuda < 2021.1

    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    model.cpu()
    model.eval()

    dummy_input = torch.rand((1, 3, 224, 224))

    output_path = os.path.join(CURRENT_PATH, 'out', 'mbv1_224.onnx')

    # Model with static shapes
    torch.onnx.export(model, dummy_input, output_path, opset_version=13)

    compare_onnx_tensorrt(
        output_path,
        USE_FP16,
        USE_INT8,
        WORKSPACE_SIZE,
    )

    # Model with dynamic shapes
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["x"],
        output_names=["out"],
        dynamic_axes={
            "x": [0],
            "out": [0],
        },
        opset_version=13,
    )

    # Build with dynamic shape and using a specific dummy input
    DYNAMIC_SHAPE_BUILD_DICT = {'x': ((1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))}
    DYNAMIC_SHAPE_EVAL_DICT = {'x': (1, 3, 224, 224)}
    INPUT_PATH_MAPPING = os.path.join(CURRENT_PATH, 'out', 'mbv1_224.bin')

    dummy_input.numpy().tofile(INPUT_PATH_MAPPING)

    compare_onnx_tensorrt(
        output_path,
        USE_FP16,
        USE_INT8,
        WORKSPACE_SIZE,
        DYNAMIC_SHAPE_BUILD_DICT,
        DYNAMIC_SHAPE_EVAL_DICT,
        INPUT_PATH_MAPPING,
    )


if __name__ == '__main__':
    main()
