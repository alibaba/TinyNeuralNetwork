import torch
import torch.quantization as torch_q
import torch.nn.functional as F
import numpy as np
from typing import Any, List, Tuple, Optional, Dict, Union
import copy


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


class HistogramObserverKL(torch_q.HistogramObserver):
    def _compute_threshold(self, distribution: np.ndarray, m_bin_number=2048) -> int:
        """Compute the quantization error using Kullback-Leibler divergence.
        We filter out outliers in input distribution by searching the threshold for minimizing KL-Divergence.
        Ref:https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

        Args:
            distribution: ndarray, the distribution of Calibration Sets normalized to 1.
            m_bin_number: int, the bins_number of distribution.
        Returns:
            threshold: int, the best threshold with the minimum KL.
        """
        m_bin_number = m_bin_number + 1
        target_bin_numbers = 128
        threshold = target_bin_numbers
        min_kl_divergence = float('inf')

        after_threshold_sum = np.sum(distribution[target_bin_numbers:])
        cumsum_dist = np.zeros(distribution.size + 1, dtype=distribution.dtype)
        np.cumsum(distribution, out=cumsum_dist[1:])
        cumsum_nozeros = np.zeros(distribution.size + 1, dtype=distribution.dtype)
        np.cumsum(distribution != 0, out=cumsum_nozeros[1:])
        is_nonzero_distribution = distribution != 0
        is_nonzero_distribution = np.append(is_nonzero_distribution, False)

        for i in range(target_bin_numbers, m_bin_number):
            quantize_dis = np.zeros(target_bin_numbers)
            expanded_dis = np.zeros(i)
            candidate_dis = copy.deepcopy(distribution)[:i]
            candidate_dis[i - 1] = candidate_dis[i - 1] + after_threshold_sum
            if i != m_bin_number - 1:
                after_threshold_sum -= distribution[i]
            else:
                after_threshold_sum = np.zeros(1)

            bin_interval = i / target_bin_numbers

            # merge i bins to target bins
            j_ = np.arange(target_bin_numbers)
            start_ = j_ * bin_interval
            end_ = start_ + bin_interval

            left_upper_ = np.ceil(start_).astype('int32')
            right_lower_ = np.floor(end_).astype('int32')

            left_flag = left_upper_ > start_
            right_flag = right_lower_ < end_

            left_scale_ = left_upper_ - start_
            right_scale_ = end_ - right_lower_

            quantize_dis[left_flag] += left_scale_[left_flag] * distribution[left_upper_[left_flag] - 1]
            quantize_dis[right_flag] += right_scale_[right_flag] * distribution[right_lower_[right_flag]]
            quantize_dis += cumsum_dist[right_lower_] - cumsum_dist[left_upper_]

            # expand target bins to i bins
            count_ = np.zeros(target_bin_numbers)
            count_[left_flag] += left_scale_[left_flag] * is_nonzero_distribution[left_upper_[left_flag] - 1]
            count_[right_flag] += right_scale_[right_flag] * is_nonzero_distribution[right_lower_[right_flag]]
            count_ += cumsum_nozeros[right_lower_] - cumsum_nozeros[left_upper_]

            to_expand_value_ = np.zeros(target_bin_numbers)
            count_flag = count_ != 0
            to_expand_value_[count_flag] = quantize_dis[count_flag] / count_[count_flag]
            left_expand_flag = np.logical_and(count_flag, left_flag, is_nonzero_distribution[left_upper_ - 1])
            expanded_dis[left_upper_[left_expand_flag] - 1] += (
                to_expand_value_[left_expand_flag] * left_scale_[left_expand_flag]
            )
            right_expand_flag = np.logical_and(count_flag, right_flag, is_nonzero_distribution[right_lower_])
            expanded_dis[right_lower_[right_expand_flag]] += (
                to_expand_value_[right_expand_flag] * right_scale_[right_expand_flag]
            )

            k = np.floor(bin_interval).astype('int32')
            last_flag = k - (right_lower_ - left_upper_) == 0
            for m in range(right_lower_[0] - 1):
                expanded_dis[left_upper_ + m] += to_expand_value_ * is_nonzero_distribution[left_upper_ + m]
            expanded_dis[left_upper_ + right_lower_[0] - 1] += (
                to_expand_value_ * last_flag * is_nonzero_distribution[left_upper_ + right_lower_[0] - 1]
            )

            # Calculate the Kl divergence of expanded_dis and candidate_dis
            expanded_dis = torch.from_numpy(expanded_dis)
            candidate_dis = torch.from_numpy(candidate_dis)
            curKL = F.kl_div(expanded_dis.log(), candidate_dis, reduction='sum')
            if curKL < min_kl_divergence and curKL != 1.0:
                min_kl_divergence = curKL
                threshold = i

        return threshold

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search for optimal cutoff range for asymmetric quantization.
        First, search for right_threshold, then search from right to left for left_threshold by minimizing the KL-d.
        """
        bins = self.histogram.clone()
        bins[0] = bins[1]
        bins_np = bins.numpy()
        bins_np[bins_np < 0] = 0
        total_num = np.sum(bins_np)
        bins_np = bins_np / total_num
        bin_width = (self.max_val - self.min_val) / self.bins

        right_threshold = self._compute_threshold(bins_np, 2048)
        left_dis = bins_np[:right_threshold]
        left_dis = left_dis[::-1]
        m_bin_number = left_dis.size
        left_threshold_ = self._compute_threshold(left_dis, m_bin_number=m_bin_number)
        left_threshold = right_threshold - left_threshold_
        new_min = self.min_val + bin_width * left_threshold
        new_max = self.min_val + bin_width * right_threshold

        return new_min, new_max
