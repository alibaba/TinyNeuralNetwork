import torch
import torch.quantization as torch_q


class MinMaxObserver(torch_q.MinMaxObserver):
    def __init__(self, *args, **kwargs) -> None:
        super(MinMaxObserver, self).__init__(*args, **kwargs)
        self.quant_min = -127
        self.quant_max = 127


class PerChannelMinMaxObserver(torch_q.PerChannelMinMaxObserver):
    def __init__(self, *args, **kwargs) -> None:
        super(PerChannelMinMaxObserver, self).__init__(*args, **kwargs)
        self.quant_min = -127
        self.quant_max = 127


class MovingAverageMinMaxObserver(torch_q.MovingAverageMinMaxObserver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min = -127
        self.quant_max = 127


class MovingAveragePerChannelMinMaxObserver(torch_q.MovingAveragePerChannelMinMaxObserver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min = -127
        self.quant_max = 127
