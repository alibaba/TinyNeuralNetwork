import argparse
import glob
import os
import re
import subprocess
import sys
import time

from pprint import pprint

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter
from tinynn.converter.utils.tflite import parse_model


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


def glob_paths(path):
    paths = []
    fn, ext = os.path.splitext(path)
    patterns = [f'{fn}_{suffix}_*{ext}' for suffix in ('float', 'dq')]
    for pat in patterns:
        paths.extend(glob.glob(pat))
    return paths


def benchmark_model(path, count=50, warmup=1):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(path, num_threads=1)
    interpreter.allocate_tensors()

    for _ in range(warmup):
        interpreter.invoke()

    st = time.time()
    for _ in range(count):
        interpreter.invoke()
    et = time.time()

    return (et - st) / count * 1000


def benchmark_model_adb(path):
    device_dir = '/data/local/tmp'
    device_path = os.path.join(device_dir, os.path.basename(path))
    benchmark_app = 'android_arm_benchmark_model'
    benchmark_path = os.path.join(device_dir, benchmark_app)

    subprocess.call(['adb', 'push', path, device_dir])
    run_out = subprocess.check_output(
        ['adb', 'shell', f'{benchmark_path} --graph={device_path} --num_threads=1'], stderr=subprocess.DEVNULL
    ).decode()
    subprocess.call(['adb', 'shell', f'rm -f {device_path}'])

    times = re.findall('Inference \(avg\): (.*)', run_out)
    return float(times[0]) / 1000


def main_worker(args):
    model = SimpleLSTM(args.input_size, args.hidden_size, args.num_layers, args.num_classes, args.bidirectional)

    # Provide a viable input for the model
    dummy_input = torch.rand((args.steps, args.batch_size, args.input_size))

    print(model)

    tflite_path = 'out/dynamic_quant_model.tflite'

    with torch.no_grad():
        model.eval()
        model.cpu()

        # The code section below is used to convert the model to the TFLite format
        converter = TFLiteConverter(
            model,
            dummy_input,
            tflite_path=tflite_path,
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
            # Generate single op models for hybrid quantizable ops
            hybrid_gen_single_op_models=True,
            # Enable rewrite for BidirectionLSTMs to UnidirectionalLSTMs
            map_bilstm_to_lstm=False,
        )

        converter.convert()

    f_timings = {}
    dq_timings = {}

    prefix = os.path.splitext(tflite_path)[0]
    paths = glob_paths(tflite_path)
    for path in paths:
        # By default, benchmark runs on Android.
        # If you want to do that on native Python, you may switch to use `benchmark_model` instead
        tm = benchmark_model_adb(path)
        # tm = benchmark_model(path)

        var, loc = os.path.splitext(path)[0][len(prefix) + 1 :].split('_')
        loc = int(loc)

        if var == 'float':
            f_timings[loc] = tm
        else:
            dq_timings[loc] = tm

    tfl_model = parse_model(tflite_path)

    print('Timings:')
    hybrid_config = {}
    for k in f_timings:
        op_out = tfl_model.Subgraphs(0).Operators(k).Outputs(0)
        name = tfl_model.Subgraphs(0).Tensors(op_out).Name().decode()
        print(f'layer {k}("{name}"): {f_timings[k]:.2f}ms (float) vs {dq_timings[k]:.2f}ms (dq)')
        hybrid_config[name] = f_timings[k] > dq_timings[k]

    print('Hybrid config:')
    pprint(hybrid_config)

    # Fallback when floating point kernels are faster
    if not all(hybrid_config.values()):
        time_diff = sum((0.0 if f_timings[k] > dq_timings[k] else dq_timings[k] - f_timings[k] for k in f_timings))
        print(f'Partial fallback saves {time_diff:.2f}ms')
        converter = TFLiteConverter(
            model,
            dummy_input,
            tflite_path=tflite_path,
            strict_symmetric_check=converter.strict_symmetric_check,
            quantize_target_type=converter.q_type.__name__,
            # Enable hybrid quantization
            hybrid_quantization_from_float=converter.hybrid,
            # Enable hybrid per-channel quantization (lower q-loss, but slower)
            hybrid_per_channel=converter.hybrid_per_channel,
            # Use asymmetric inputs for hybrid quantization (probably lower q-loss, but a bit slower)
            hybrid_asymmetric_inputs=converter.hybrid_asymmetric_inputs,
            # Enable hybrid per-channel quantization for `Conv2d` and `DepthwiseConv2d`
            hybrid_conv=converter.hybrid_conv,
            # Hybrid configurations
            hybrid_config=hybrid_config,
            # Enable rewrite for BidirectionLSTMs to UnidirectionalLSTMs
            map_bilstm_to_lstm=False,
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
