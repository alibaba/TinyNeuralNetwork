import unittest


class ImportTester(unittest.TestCase):
    def test_oneshot_pruner(self):
        from tinynn.prune.oneshot_pruner import OneShotChannelPruner

    def test_admm_pruner(self):
        from tinynn.prune.admm_pruner import ADMMPruner

    def test_netadapt_pruner(self):
        from tinynn.prune.netadapt_pruner import NetAdaptPruner

    def test_tracer(self):
        from tinynn.graph.tracer import model_tracer, trace, import_patcher

    def test_qat(self):
        from tinynn.graph.quantization.quantizer import QATQuantizer

    def test_ptq(self):
        from tinynn.graph.quantization.quantizer import PostQuantizer

    def test_dyn_q(self):
        from tinynn.graph.quantization.quantizer import DynamicQuantizer

    def test_bf16(self):
        from tinynn.graph.quantization.quantizer import BF16Quantizer

    def test_converter(self):
        from tinynn.converter import TFLiteConverter

    def test_cifar10_util(self):
        from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate, calibrate

    def test_train_util(self):
        from tinynn.util.train_util import DLContext, get_device, train, get_module_device

    def test_observer(self):
        from tinynn.graph.quantization.observer import HistogramObserverKL

    def test_bn_restore(self):
        from tinynn.util.bn_restore import model_restore_bn

    def test_cle(self):
        from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize


if __name__ == '__main__':
    unittest.main()
