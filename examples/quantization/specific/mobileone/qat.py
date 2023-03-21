import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../../'))

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

yaml_ = yaml.YAML()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as torch_q

from examples.quantization.specific.util.imagenet_util import get_dataloader, train_one_epoch, validate, calibrate
from mobileone_origin import get_model

from tinynn.util.train_util import DLContext, train, get_device
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize
from tinynn.util.bn_restore import model_restore_bn
from tinynn.util.quantization_analysis_util import graph_error_analysis, layer_error_analysis


def qat_mobileone(args):
    device = get_device()
    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dummy_input = torch.rand((1, 3, 224, 224))
    dummy_real_input = next(iter(context.train_loader))[0][:1]
    print(dummy_real_input.size())
    model = get_model(variant='s1', model_pth=args.model_pth_dir)

    # When the weight distributions fluctuates greatly, CLE may significantly increase the quantization accuracy.
    if args.cle:
        model = cross_layer_equalize(model, dummy_input, device, cle_iters=4, hba_flag=False)

    # Perform BatchNorm restore after CLE to make QAT more stable and faster.
    model.train()
    if args.bn_restore:
        context.max_iteration = 100
        model = model_restore_bn(model, device, calibrate, context, False)

    with model_tracer():
        # More information for PostQuantizer initialization, see `examples/quantization/qat.py`.
        quantizer = QATQuantizer(model, dummy_input, work_dir='out', config={'ignore_layerwise_config': True})
        qat_model = quantizer.quantize()

    if torch.cuda.device_count() > 1:
        qat_model = nn.DataParallel(qat_model)

    qat_model.to(device=device)

    context.max_iteration = 10
    calibrate(qat_model, context, True)

    context.max_iteration = None

    # Use training config from TorchVision
    context.max_epoch = 30
    context.criterion = nn.CrossEntropyLoss()
    context.optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    context.scheduler = optim.lr_scheduler.StepLR(context.optimizer, step_size=10, gamma=0.1)

    context.print_freq = 100
    print(qat_model)
    qat_model.train()
    qat_model.apply(torch_q.enable_fake_quant)
    qat_model.apply(torch_q.enable_observer)
    train(qat_model, context, train_one_epoch, validate, qat=True)

    # Perform quantization error analysis with real dummy input
    dummy_input_real = next(iter(context.train_loader))[0][:1]

    # The error is accumulated by directly giving the difference in layer output
    # between the quantized model and the floating model. If you want a quantized model with high accuracy,
    # the layers closest to the final output should be less than 10%, which means the final
    # layer's cosine similarity should be greater than 90%.
    graph_error_analysis(qat_model, dummy_input_real, metric='cosine')

    # We quantize each layer separately and compare the difference
    # between the original output and the output with quantization error for each layer,
    # which is used to calculate the quantization sensitivity of each layer.
    layer_error_analysis(qat_model, dummy_input_real, metric='cosine')

    # validate the model with quantization error via fake quantization
    qat_model.apply(torch_q.disable_observer)
    validate(qat_model, context)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        if isinstance(qat_model, nn.DataParallel):
            qat_model = qat_model.module

        qat_model = torch.quantization.convert(qat_model)
        context.device = torch.device('cpu')

        # validate the quantized model
        validate(qat_model, context)

        torch.backends.quantized.engine = quantizer.backend
        converter = TFLiteConverter(model, dummy_input, tflite_path='out/mobileone_qat.tflite')
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/imagenet", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cle', type=bool, default=True)
    parser.add_argument('--bn-restore', type=bool, default=True)
    parser.add_argument('--model-pth-dir', type=str, default='out')

    args = parser.parse_args()
    qat_mobileone(args)
