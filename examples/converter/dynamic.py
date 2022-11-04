import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter


class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, out_dim, layers)
        self.fc = torch.nn.Linear(out_dim, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        out = self.fc(out)
        out = self.relu(out)
        return out


def main_worker(args):
    model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes)

    # Provide a viable input for the model
    dummy_input = torch.rand((args.steps, args.batch_size, args.input_size))

    print(model)

    with torch.no_grad():
        model.eval()
        model.cpu()

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(
            model,
            dummy_input,
            tflite_path='out/dynamic_quant_model.tflite',
            strict_symmetric_check=True,
            quantize_target_type='int8',
            # Enable hybrid quantization
            hybrid_quantization_from_float=True,
            # Enable hybrid per-channel quantization (lower q-loss, but slower)
            hybrid_per_channel=False,
            # Use asymmetric inputs for hybrid quantization (probably lower q-loss, but a bit slower)
            hybrid_asymmetric_inputs=True,
            # Enable hybrid per-channel quantization for `Conv2d` and `DepthwiseConv2d`
            hybrid_conv=True,
        )
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--input-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=10)

    args = parser.parse_args()
    main_worker(args)
