import unittest

import torch
import torch.nn as nn
from tinynn.graph.quantization.modules import QPReLU, QSiLU


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


if __name__ == '__main__':
    unittest.main()
