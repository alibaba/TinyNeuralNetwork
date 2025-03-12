import unittest
import random

import torch
import torch.nn as nn
from tinynn.graph.quantization.modules import QGLU, QPReLU, QSiLU, QLayerNorm, QRMSNorm


class QATModuleTester(unittest.TestCase):
    def test_prelu(self):
        for i in range(100):
            val = torch.rand([])
            orig = nn.PReLU(init=val)
            quant = QPReLU(orig)

            inp = (torch.randn((100, 100)) - 0.5) * 2

            orig_outp = orig(inp)
            quant_outp = quant(inp)

            if not torch.allclose(orig_outp, quant_outp):
                print('original:')
                print(orig_outp)
                print('quanted:')
                print(quant_outp)

                print('diff (min, max):', torch.max(quant_outp - orig_outp), torch.min(quant_outp - orig_outp))

                self.assertTrue(False)

    def test_prelu_multi_channel(self):
        for i in range(100):
            val = torch.rand([])
            orig = nn.PReLU(num_parameters=32, init=val)
            quant = QPReLU(orig)

            inp = (torch.randn((1, 32, 24, 24)) - 0.5) * 2

            orig_outp = orig(inp)
            quant_outp = quant(inp)

            if not torch.allclose(orig_outp, quant_outp):
                print('original:')
                print(orig_outp)
                print('quanted:')
                print(quant_outp)

                print('diff (min, max):', torch.max(quant_outp - orig_outp), torch.min(quant_outp - orig_outp))

                self.assertTrue(False)

    @unittest.skipIf(not hasattr(nn, 'SiLU'), 'nn.SiLU is not available')
    def test_silu(self):
        for i in range(100):
            orig = nn.SiLU()
            quant = QSiLU(orig)

            inp = (torch.randn((100, 100)) - 0.5) * 2

            orig_outp = orig(inp)
            quant_outp = quant(inp)

            if not torch.allclose(orig_outp, quant_outp):
                print('original:')
                print(orig_outp)
                print('quanted:')
                print(quant_outp)

                print('diff (min, max):', torch.max(quant_outp - orig_outp), torch.min(quant_outp - orig_outp))

                self.assertTrue(False)

    def test_glu(self):
        for i in range(100):
            orig = nn.GLU()
            quant = QGLU(orig)

            inp = (torch.randn((100, 100)) - 0.5) * 2

            orig_outp = orig(inp)
            quant_outp = quant(inp)

            if not torch.allclose(orig_outp, quant_outp):
                print('original:')
                print(orig_outp)
                print('quanted:')
                print(quant_outp)

                print('diff (min, max):', torch.max(quant_outp - orig_outp), torch.min(quant_outp - orig_outp))

                self.assertTrue(False)

    def test_layer_norm(self):
        for i in range(100):
            normalized_shape = tuple(random.randint(10, 100) for _ in range(random.randint(1, 3)))
            non_normalized_shape = tuple(random.randint(1, 100) for _ in range(random.randint(1, 2)))

            orig = nn.LayerNorm(normalized_shape)
            quant = QLayerNorm(orig)

            inp = torch.randn((*non_normalized_shape, *normalized_shape))

            orig_outp = orig(inp)
            quant_outp = quant(inp)

            if not torch.allclose(orig_outp, quant_outp, atol=1e-6):
                print(normalized_shape, non_normalized_shape)
                print('original:')
                print(orig_outp)
                print('quanted:')
                print(quant_outp)

                print('diff (min, max):', torch.max(quant_outp - orig_outp), torch.min(quant_outp - orig_outp))

                self.assertTrue(False)

    @unittest.skipIf(not hasattr(torch.nn, 'RMSNorm'), 'RMSNorm is not supported')
    def test_rms_norm(self):
        for i in range(100):
            normalized_shape = tuple(random.randint(10, 100) for _ in range(random.randint(1, 3)))
            non_normalized_shape = tuple(random.randint(1, 100) for _ in range(random.randint(1, 2)))

            orig = nn.RMSNorm(normalized_shape)
            quant = QRMSNorm(orig)

            inp = torch.randn((*non_normalized_shape, *normalized_shape))

            orig_outp = orig(inp)
            quant_outp = quant(inp)

            if not torch.allclose(orig_outp, quant_outp, atol=1e-6):
                print(normalized_shape, non_normalized_shape)
                print('original:')
                print(orig_outp)
                print('quanted:')
                print(quant_outp)

                print('diff (min, max):', torch.max(quant_outp - orig_outp), torch.min(quant_outp - orig_outp))

                self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
