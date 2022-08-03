import os
import sys

import numpy as np
import tensorflow as tf
import torch

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter
from tinynn.converter.base import GraphOptimizer


def main_worker():
    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    model.cpu()
    model.eval()

    dummy_input = torch.rand((1, 3, 224, 224))

    output_path = os.path.join(CURRENT_PATH, 'out', 'mbv1_224.tflite')

    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = 'qnnpack'

    # The code section below is used to convert the model to the TFLite format
    # When `preserve_tensors=True` is specified, the intermediate tensors will be preserved,
    # so that they can be compared with those generated with other backends.
    # You may also need to tune the optimize level to adjust the granularity of the comparison.
    # For example, using values like `GraphOptimizer.FOLD_BUFFER` or `GraphOptimizer.NO_OPTIMIZE`
    # will ensure comparsion of the outputs in almost every layer,
    # while with `GraphOptimizer.ALL_OPTIMIZE` or `GraphOptimizer.COMMON_OPTIMIZE`,
    # some intermediate layers will be skipped because they may be fused with other layers.
    converter = TFLiteConverter(
        model, dummy_input, output_path, preserve_tensors=True, optimize=GraphOptimizer.ALL_OPTIMIZE
    )
    converter.convert()

    # Flag variable whether you want to compare the output tensors or all the intermediate tensors
    # The suggestion is to use layerwise comparison only when the outputs don't match.
    layerwise = False

    # As for layerwise comparison, we need to pass `experimental_preserve_all_tensors=True`,
    # which requires `tensorflow >= 2.5.0`.
    tfl_interpreter_args = {'model_path': output_path}
    if layerwise:
        tfl_interpreter_args['experimental_preserve_all_tensors'] = True

    # Initialize TFLite interpreter
    interpreter = tf.lite.Interpreter(**tfl_interpreter_args)
    interpreter.allocate_tensors()

    # Get input and output tensors from the TFLite interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if layerwise:
        details = interpreter.get_tensor_details()
    else:
        details = output_details

    tfl_tensor_idx_map = {t['name']: t['index'] for t in details}

    # Prepare inputs for the TFLite interpreter
    tfl_inputs = converter.inputs_for_tflite()
    for i, t in enumerate(tfl_inputs):
        interpreter.set_tensor(input_details[i]['index'], t)

    transpose_names = set()
    io_details = list(input_details) + list(output_details)
    io_transpose = list(converter.common_graph.input_transpose) + list(converter.common_graph.output_transpose)
    for d, t in zip(io_details, io_transpose):
        if t:
            transpose_names.add(d['name'])

    # Inference
    interpreter.invoke()

    # Get common nodes
    torch_names = converter.tensor_names()
    common_names = set(torch_names).intersection(set(tfl_tensor_idx_map))

    atol = 1e-5
    rtol = 1e-3

    for n in torch_names:
        if n not in common_names:
            continue

        # Get outputs from the backends
        tfl_v = interpreter.get_tensor(tfl_tensor_idx_map[n])
        torch_v = converter.get_value(n)

        # Convert the PyTorch tensor to a NumPy array
        if torch_v.dtype in (torch.quint8, torch.qint8):
            torch_v = torch_v.dequantize().numpy() / torch_v.q_scale() + torch_v.q_zero_point()
        else:
            torch_v = torch_v.numpy()

        # Align shapes and dtypes of the tensors
        if n in transpose_names:
            torch_v = np.transpose(torch_v, (0, 2, 3, 1))

        if torch_v.dtype != tfl_v.dtype:
            tfl_v = tfl_v.astype(torch_v.dtype)

        # Compare the tensors using `np.allclose`
        matches = np.allclose(tfl_v, torch_v, rtol=rtol, atol=atol)
        print(f'Output {n} value matches: {matches}')

        # Calculate absolute difference
        diff = np.abs(torch_v - tfl_v)

        diff_mean = np.mean(diff)
        diff_min = np.min(diff)
        diff_max = np.max(diff)

        abs_err_percent = np.mean((diff > atol).astype('float32')) * 100
        print(
            f'Output {n} absolute difference min,mean,max: {diff_min},{diff_mean},{diff_max} (error:'
            f' {abs_err_percent:.2f}%)'
        )

        # Calculate relative difference
        torch_v_nonzero = (torch_v != 0).astype('bool')
        if np.all(~torch_v_nonzero):
            rel_err = np.array([float('inf')] * len(torch_v))
        else:
            rel_err = diff[torch_v_nonzero] / np.abs(torch_v[torch_v_nonzero])

        rel_diff_mean = np.mean(rel_err)
        rel_diff_min = np.min(rel_err)
        rel_diff_max = np.max(rel_err)

        rel_err_percent = np.mean((rel_err > rtol).astype('float32')) * 100
        print(
            f'Output {n} relative difference min,mean,max: {rel_diff_min},{rel_diff_mean},{rel_diff_max} (error:'
            f' {rel_err_percent:.2f}%)'
        )


if __name__ == '__main__':
    main_worker()
