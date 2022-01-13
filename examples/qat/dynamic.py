import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tinynn.graph.tracer import model_tracer, trace
from tinynn.graph.quantization.quantizer import DynamicQuantizer
from models.cifar10.mobilenet import Mobilenet, DEFAULT_STATE_DICT
from tinynn.converter import TFLiteConverter


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


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
    with model_tracer():
        model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes)

        # Provide a viable input for the model
        dummy_input = torch.rand((args.steps, args.batch_size, args.input_size))

        # TinyNeuralNetwork provides a DynamicQuantizer class that performs dynamic quantization
        # The model returned by the `quantize` function is already dynamically quantized
        quantizer = DynamicQuantizer(model, dummy_input, work_dir='out', config={'asymmetric': False})
        qat_model = quantizer.quantize()

    print(qat_model)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(qat_model, dummy_input,
                                    tflite_path='out/dynamic_quant_model.tflite',
                                    asymmetric=False,
                                    quantize_target_type='int8')
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
