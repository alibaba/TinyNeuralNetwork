import os
import torch

from tinynn.graph.rewriter import rewrite_for_tensorrt_export

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm([5, 10, 10])

    def forward(self, x):
        return self.norm(x)


model = Model()
model.eval()

# Perform some useful rewrites, including transformations like nn.LayerNorm to tensorrt.LayerNorm
# Those rewrites require TensorRT >= 8.5
rewrite_for_tensorrt_export(model)

output_path = os.path.join(CURRENT_PATH, 'out', 'test.onnx')
torch.onnx.export(model, torch.randn(20, 5, 10, 10), output_path)
