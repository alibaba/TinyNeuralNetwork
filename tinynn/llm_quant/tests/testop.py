import unittest

import torch
from easyquant import (
    decompress_int4,
    decompress_int8,
    quantize_per_token,
    gemm,
    dequantize_bias_per_token,
    dequantize_per_token,
)

batch_seq = 128
in_fea = 4096
out_fea = 4096 * 4


class TestOps(unittest.TestCase):
    def test_gemm_cuda(self):
        tensor1 = torch.randint(-128, 127, (batch_seq, in_fea), dtype=torch.int8).cuda()
        tensor2 = torch.randint(-128, 127, (out_fea, in_fea), dtype=torch.int8).cuda()

        actual = torch.empty((tensor1.shape[0], out_fea), dtype=torch.int32, device=torch.device('cuda'))

        gemm(actual, tensor1, tensor2)
        expected = torch.nn.functional.linear(
            tensor1.cpu().to(torch.int32), tensor2.cpu().to(torch.int32), bias=None
        ).cuda()

        torch.testing.assert_close(actual, expected)

    def test_gemm_single_batch_cuda(self):
        batch_seq = 1
        tensor1 = torch.randint(-128, 127, (batch_seq, in_fea), dtype=torch.int8).cuda()
        tensor2 = torch.randint(-128, 127, (out_fea, in_fea), dtype=torch.int8).cuda()

        actual = torch.empty((tensor1.shape[0], out_fea), dtype=torch.int32, device=torch.device('cuda'))

        gemm(actual, tensor1, tensor2)
        expected = torch.mm(
            tensor1.cpu().to(dtype=torch.int32),
            tensor2.cpu().transpose(0, 1).to(dtype=torch.int32),
        ).cuda()

        torch.testing.assert_close(actual, expected)

    # this test will fail because there is some calculation error between torch and cuda which will not exceed 1.
    def test_quantize_per_token_cuda(self):
        tensor1 = torch.randn(batch_seq, out_fea).to(dtype=torch.float16).cuda()
        device = tensor1.device
        actual_tensor_q = torch.empty_like(tensor1, dtype=torch.int8, device=device)
        actual_scale = torch.zeros(batch_seq, device=device)

        quantize_per_token(actual_tensor_q, tensor1, actual_scale)

        ref_scale = tensor1.to(torch.float32).abs().max(dim=-1).values / 127.0
        ref_out = torch.clamp(torch.round(tensor1.to(torch.float32) / ref_scale[:, None]), -127, 127).to(
            dtype=torch.int8
        )

        torch.testing.assert_close(actual_scale, ref_scale)
        torch.testing.assert_close(actual_tensor_q, ref_out)

    def test_dequantze_token_cuda(self):
        tensor1 = torch.randint(-128, 128, (batch_seq, out_fea), dtype=torch.int32).cuda()
        weight_scale = torch.rand(out_fea).cuda()
        input_scale = torch.rand(batch_seq).cuda()
        out = torch.empty(batch_seq, out_fea, dtype=torch.float16).cuda()

        dequantize_per_token(out, tensor1, input_scale, weight_scale)
        ref_out = (tensor1.to(dtype=torch.float32) * (weight_scale * input_scale.view(-1, 1))).to(dtype=torch.float16)

        torch.testing.assert_close(out, ref_out)

    def test_dequantize_bias_token_cuda(self):
        tensor1 = torch.randint(-128, 128, (batch_seq, out_fea), dtype=torch.int32).cuda()
        weight_scale = torch.rand(out_fea).cuda()
        input_scale = torch.rand(batch_seq).cuda()
        bias = torch.rand(out_fea).to(dtype=torch.float16).cuda()
        out = torch.empty(batch_seq, out_fea, dtype=torch.float16).cuda()

        dequantize_bias_per_token(out, tensor1, input_scale, weight_scale, bias)
        ref_out = (tensor1.to(dtype=torch.float32) * (weight_scale * input_scale.view(-1, 1)) + bias.float()).to(
            dtype=torch.float16
        )

        torch.testing.assert_close(out, ref_out)

    def test_decompress_int4_cuda(self):
        tensor = torch.randint(-8, 7, (in_fea, out_fea), dtype=torch.int8).cuda()
        scale = torch.rand(in_fea).cuda()

        packed = ((tensor[:, ::2] & 0b00001111) << 4) | (tensor[:, 1::2] & 0b00001111)

        actual = torch.empty_like(tensor, dtype=torch.half, device=torch.device('cuda'))

        decompress_int4(actual, packed, scale)
        expected = (
            torch.stack((packed >> 4, packed << 4 >> 4), -1).view(in_fea, -1).to(dtype=torch.float32) * scale[:, None]
        ).to(dtype=torch.half)

        torch.testing.assert_close(actual, expected)

    def test_decompress_int8_cuda(self):
        tensor = torch.randint(-128, 127, (in_fea, out_fea), dtype=torch.int8).cuda()
        scale = torch.rand(in_fea).cuda()

        actual = torch.empty_like(tensor, dtype=torch.half, device=torch.device('cuda'))

        decompress_int8(actual, tensor, scale)
        expected = (tensor.to(dtype=torch.float32) * scale[:, None]).to(dtype=torch.half)

        torch.testing.assert_close(actual, expected)

    # TODO
    def test_compress_int4_cuda(self):
        pass

    def test_compress_int8_cuda(self):
        pass

    def test_dequantize_int8_cuda(self):
        pass

    def test_dequantize_int8_bias_cuda(self):
        pass


if __name__ == "__main__":
    unittest.main()
