import argparse
import os

import torch
from tinynn.converter import TFLiteConverter
from tinynn.util.converter_util import export_converter_files, parse_config

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def export_files():
    from models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet

    model = Mobilenet()
    model.load_state_dict(torch.load(DEFAULT_STATE_DICT))
    model.cpu()
    model.eval()

    dummy_input = torch.rand((1, 3, 224, 224))

    export_dir = 'out'
    export_name = 'mbv1_224'

    export_converter_files(model, dummy_input, export_dir, export_name)

    json_file = os.path.join(CURRENT_PATH, export_dir, f'{export_name}.json')
    return json_file


def main_worker(args):
    json_file = args.path
    if json_file is None:
        json_file = export_files()

    # We will try to parse the config and prepare the inputs for you.
    # If you want to use your own inputs, just assign it to `generated_inputs` here.
    torch_model_path, tflite_model_path, input_transpose, generated_inputs, output_transpose = parse_config(json_file)

    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = 'qnnpack'

    with torch.no_grad():
        model = torch.jit.load(torch_model_path)
        model.cpu()
        model.eval()

        # Pay attention to the arguments `input_transpose` and `output_transpose` in the next line.
        # By default, we will perform nchw -> nhwc transpose every 4D input and output tensor.
        # If you don't want to do this, please pass in False for them.
        converter = TFLiteConverter(model, generated_inputs, tflite_model_path, input_transpose, output_transpose)
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', metavar='DIR', default=None, help='path to the config (.json)')

    args = parser.parse_args()
    main_worker(args)
