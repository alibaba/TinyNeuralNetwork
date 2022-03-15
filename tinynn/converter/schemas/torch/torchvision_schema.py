from abc import abstractmethod

from ...operators.torch.base import OperatorConverter


class TorchVisionPsRoiAlignSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::ps_roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> (Tensor, Tensor)'''
        pass


class TorchVisionRoiAlignSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> (Tensor)'''
        pass


class TorchVisionPsRoiPoolSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::ps_roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)'''
        pass


class TorchVisionDeformConv2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor)'''
        pass


class TorchVisionInterpolateBilinear2dAaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::_interpolate_bilinear2d_aa(Tensor input, int[] output_size, bool align_corners) -> (Tensor)'''
        pass


class TorchVisionInterpolateBicubic2dAaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::_interpolate_bicubic2d_aa(Tensor input, int[] output_size, bool align_corners) -> (Tensor)'''
        pass


class TorchVisionNmsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> (Tensor)'''
        pass


class TorchVisionRoiPoolSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''torchvision::roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)'''
        pass
