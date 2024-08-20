import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer


class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers, num_classes, bidirectional):
        super(SimpleLSTM, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.lstm = torch.nn.LSTM(in_dim, out_dim, layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(out_dim * num_directions, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        out = self.fc(out)
        out = self.relu(out)
        return out


def main_worker(args):
    model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes, args.bidirectional)

    # Provide a viable input for the model
    dummy_input = torch.rand((args.steps, args.batch_size, args.input_size))

    # Please see 'ptq.py' for more details for using PostQuantizer.
    quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'quantize_op_action': {nn.LSTM: 'rewrite'}})
    ptq_model = quantizer.quantize()

    print(ptq_model)

    for _ in range(5):
        ptq_model(torch.rand_like(dummy_input))

    with torch.no_grad():
        ptq_model.eval()
        ptq_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        ptq_model = quantizer.convert(ptq_model)

        print(ptq_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = quantizer.backend

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(
            ptq_model,
            dummy_input,
            tflite_path='out/ptq_with_dynamic_quant_lstm_model.tflite',
            quantize_target_type='int8',
            rewrite_quantizable=True,
            # Enable hybrid quantization
            hybrid_quantization_from_float=True,
            # Enable hybrid per-channel quantization (lower q-loss, but slower)
            hybrid_per_channel=False,
            # Use asymmetric inputs for hybrid quantization (probably lower q-loss, but a bit slower)
            hybrid_asymmetric_inputs=False,
            # Enable int16 hybrid lstm quantization
            hybrid_int16_lstm=True,
            # Enable rewrite for BidirectionLSTMs to UnidirectionalLSTMs
            map_bilstm_to_lstm=True,
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
    parser.add_argument('--bidirectional', action='store_true')

    args = parser.parse_args()
    main_worker(args)
