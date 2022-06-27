import io
import os
import sys

import torch

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from tinynn.converter import TFLiteConverter


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
    # If you want perform dynamic quantization on the float models,
    # you may pass the following arguments.
    #   `quantize_target_type='int8', hybrid_quantization_from_float=True, hybrid_per_channel=False`
    # As for static quantization (e.g. quantization-aware training and post-training quantization),
    # please refer to the code examples in the `examples/quantization` folfer.
    converter = TFLiteConverter(model, dummy_input, output_path)
    converter.convert()


if __name__ == '__main__':
    main_worker()
