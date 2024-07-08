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


def _quantize_lstm_weight(model: nn.LSTM, asym=False, eps=torch.tensor([1e-6])):
    '''Quantize the weights of LSTM individually to align with TFLite's implementation.'''

    def fake_quantize(weight, asym=False, bit=8, eps=torch.tensor([1e-6])):
        assert bit == 8
        if weight.numel() == 0:
            return weight
        quant_min, quant_max = -127, 127
        device = weight.device
        weight_shape = list(weight.size())
        weight_parts = weight.split(weight_shape[0] // 4, dim=0)
        weight_quant_parts = []
        for i in range(4):
            min_val, max_val = torch.aminmax(weight_parts[i])
            zero_point = torch.zeros(min_val.size(), dtype=torch.int64, device=device)
            if not asym:
                max_val_pos = torch.max(-min_val, max_val)
                scale = max_val_pos / (float(quant_max - quant_min) / 2)
                scale = torch.max(scale, eps)
            else:
                scale = (max_val - min_val) / float(quant_max - quant_min)
                scale = torch.max(scale, eps)
                zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
                zero_point = torch.clamp(zero_point, quant_min, quant_max)

            # do fake quantize
            weight_quant_parts.append(
                torch.fake_quantize_per_tensor_affine(weight_parts[i], scale, zero_point, quant_min, quant_max)
            )
        weight_quant = torch.cat(weight_quant_parts)
        return weight_quant

    weight_ih_l0_quant = fake_quantize(model.weight_ih_l0, asym=asym, bit=8, eps=eps)
    weight_hh_l0_quant = fake_quantize(model.weight_hh_l0, asym=asym, bit=8, eps=eps)
    setattr(model, 'weight_ih_l0_og', model.weight_ih_l0.clone())
    setattr(model, 'weight_hh_l0_og', model.weight_hh_l0.clone())
    model.weight_ih_l0.data.copy_(weight_ih_l0_quant)
    model.weight_hh_l0.data.copy_(weight_hh_l0_quant)


def _reset_lstm_weight(model: nn.LSTM):
    model.weight_ih_l0.data.copy_(model.weight_ih_l0_og)
    model.weight_hh_l0.data.copy_(model.weight_hh_l0_og)
    delattr(model, 'weight_ih_l0_og')
    delattr(model, 'weight_hh_l0_og')


def quantize_weight_align_tflite(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.LSTM):
                _quantize_lstm_weight(module)


def reset_lstm_weight(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.LSTM):
                _reset_lstm_weight(module)


def main_worker(args):
    model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes)

    # Provide a viable input for the model
    dummy_input = torch.rand((args.steps, args.batch_size, args.input_size))

    # Please see 'ptq.py' for more details for using PostQuantizer.
    quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={'quantize_op_action': {nn.LSTM: 'rewrite'}})
    ptq_model = quantizer.quantize()

    print(ptq_model)

    # quantize the lstm's weight before calibrating.
    quantize_weight_align_tflite(ptq_model)

    for _ in range(5):
        ptq_model(torch.rand_like(dummy_input))

    # reset the lstm's weight.
    reset_lstm_weight(ptq_model)

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
