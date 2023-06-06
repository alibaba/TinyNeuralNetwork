import copy
import unittest

import torch

import tinynn.llm_quant.modules as tinynn_m


def test_fc_quant_nobias(quant_mod, batch, seq, in_features, out_features):
    # This test may fail due to the possibility of a difference between the quantize_per_token kernel in the CUDA
    # and the PyTorch, However, this difference is likely to be small and will not significantly affect the result.
    input_tensor = torch.randn((batch, seq, in_features)).half().cuda()
    fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=None).half()

    fc_copy = copy.deepcopy(fc)
    quantized_fc_real = tinynn_m.QLinear(fc, quant_mode=quant_mod).cuda()
    q_output = quantized_fc_real(input_tensor)

    tinynn_m.SPEEDUP = False
    quantized_fc_simu = tinynn_m.QLinear(fc_copy, quant_mode=quant_mod).cuda()
    fq_output = quantized_fc_simu(input_tensor)

    torch.testing.assert_allclose(q_output, fq_output)


class TestOps(unittest.TestCase):
    def test_fc_dynamic_cuda(self):
        test_fc_quant_nobias('dynamic', batch=1, seq=128, in_features=4096, out_features=4096 * 4)

    def test_fc_weight4_cuda(self):
        test_fc_quant_nobias('weight4', batch=1, seq=128, in_features=4096, out_features=4096 * 4)

    def test_fc_weight8_cuda(self):
        test_fc_quant_nobias('weight8', batch=1, seq=128, in_features=4096, out_features=4096 * 4)

    def test_fc_non_infea_align_cuda(self):
        test_fc_quant_nobias('dynamic', batch=1, seq=128, in_features=4096 + 1, out_features=4096 * 4)

    def test_fc_non_outfea_align_cuda(self):
        test_fc_quant_nobias('dynamic', batch=1, seq=128, in_features=4096, out_features=4096 * 4 + 1)
