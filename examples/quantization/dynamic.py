import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import DynamicQuantizer
from tinynn.graph.tracer import model_tracer


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
    # Although the following example works, we strongly encourge you to use the code example under the `converter`
    # directory for dynamic quantization. The reason are listed below.
    #  (a) TFLiteConverter supports dynamic quantization for ops including `Conv2d` and `DepthwiseConv2d`, while
    #      `DynamicQuantizer` doesn't
    #  (b) `DynamicQuantizer` leads to more quantization errors for LSTMs, as the weights of all gates are view
    #       as a whole in PyTorch, but are viewed as sepeated tensors in Tensorflow.

    with model_tracer():
        model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes)

        # Provide a viable input for the model
        dummy_input = torch.rand((args.steps, args.batch_size, args.input_size))

        # TinyNeuralNetwork provides a DynamicQuantizer class that performs dynamic quantization
        # The model returned by the `quantize` function is already dynamically quantized
        quantizer = DynamicQuantizer(model, dummy_input, work_dir='out', config={'asymmetric': False})
        q_model = quantizer.quantize()

    print(q_model)

    with torch.no_grad():
        q_model.eval()
        q_model.cpu()

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(
            q_model,
            dummy_input,
            tflite_path='out/dynamic_quant_model.tflite',
            strict_symmetric_check=True,
            quantize_target_type='int8',
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
