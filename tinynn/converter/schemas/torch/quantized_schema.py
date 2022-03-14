from abc import abstractmethod

from ...operators.torch.base import OperatorConverter


class QuantizedRelu6Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::relu6(Tensor qx, bool inplace=False) -> (Tensor)'''
        pass


class QuantizedMaxPool1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::max_pool1d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor)'''
        pass


class QuantizedInstanceNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedMulScalarSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::mul_scalar(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::mul_scalar.Tensor(Tensor qa, Tensor b) -> (Tensor qc)'''
        pass


class QuantizedMulSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::mul(Tensor qa, Tensor qb, float scale, int zero_point) -> (Tensor qc)
           quantized::mul.Scalar(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::mul.Scalar2(Scalar b, Tensor qa) -> (Tensor qc)'''
        pass


class QuantizedLinearReluDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::linear_relu_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> (Tensor Y)'''
        pass


class QuantizedHardswishSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::hardswish(Tensor input, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedEmbeddingBag4bitRowwiseOffsetsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::embedding_bag_4bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> (Tensor)'''
        pass


class QuantizedEmbeddingByteSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::embedding_byte(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, bool pruned_weights=False) -> (Tensor)'''
        pass


class QuantizedEmbeddingBagByteSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::embedding_bag_byte(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> (Tensor)'''
        pass


class QuantizedCeluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::celu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1) -> (Tensor)'''
        pass


class QuantizedEluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::elu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)'''
        pass


class QuantizedConvTranspose3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv_transpose3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedConvTranspose2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv_transpose2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedConvTranspose1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv_transpose1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedConv2dReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv2d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)
           quantized::conv2d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedConv2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv2d.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)
           quantized::conv2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedConv1dReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv1d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedCatReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::cat_relu(Tensor[] qx, int dim, float? scale, int? zero_point) -> (Tensor)'''
        pass


class QuantizedCatSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::cat(Tensor[] qx, int dim, float? scale, int? zero_point) -> (Tensor)'''
        pass


class QuantizedClampSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::clamp(Tensor qx, Scalar? min=None, Scalar? max=None) -> (Tensor qy)'''
        pass


class QuantizedBatchNorm3dReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm3d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedBatchNorm3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm3d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedBatchNorm2dReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm2d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedBatchNorm2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm2d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedBatchNorm1dReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm1d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedBatchNorm1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm1d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedBatchNormReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedAddScalarSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::add_scalar(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::add_scalar.Tensor(Tensor qa, Tensor b) -> (Tensor qc)'''
        pass


class QuantizedQuantizedRnnTanhCellDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::quantized_rnn_tanh_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> (Tensor)'''
        pass


class QuantizedQuantizedLstmCellDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::quantized_lstm_cell_dynamic(Tensor input, Tensor[] hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh) -> (Tensor, Tensor)'''
        pass


class QuantizedConv1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedThresholdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::threshold(Tensor qx, Scalar threshold, Scalar value) -> (Tensor qy)'''
        pass


class QuantizedEmbeddingBagByteRowwiseOffsetsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::embedding_bag_byte_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> (Tensor)'''
        pass


class QuantizedConv3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv3d.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)
           quantized::conv3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedAddScalarReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::add_scalar_relu(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::add_scalar_relu.Tensor(Tensor qa, Tensor b) -> (Tensor qc)'''
        pass


class QuantizedLayerNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedMaxPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::max_pool2d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> (Tensor)'''
        pass


class QuantizedGroupNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedMulReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::mul_relu(Tensor qa, Tensor qb, float scale, int zero_point) -> (Tensor qc)
           quantized::mul_relu.Scalar(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::mul_relu.Scalar2(Scalar b, Tensor qa) -> (Tensor qc)'''
        pass


class QuantizedLeakyReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::leaky_relu(Tensor qx, Scalar negative_slope, bool inplace, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedSigmoidSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::sigmoid(Tensor qx, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedConv3dReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::conv3d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> (Tensor)
           quantized::conv3d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedLinearSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> (Tensor Y)'''
        pass


class QuantizedLinearDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> (Tensor Y)'''
        pass


class QuantizedBatchNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::batch_norm(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class QuantizedEmbeddingBag4bitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::embedding_bag_4bit(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> (Tensor)'''
        pass


class QuantizedQuantizedRnnReluCellDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::quantized_rnn_relu_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> (Tensor)'''
        pass


class QuantizedAddSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> (Tensor qc)
           quantized::add.Scalar(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::add.Scalar2(Scalar b, Tensor qa) -> (Tensor qc)'''
        pass


class QuantizedMulScalarReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::mul_scalar_relu(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::mul_scalar_relu.Tensor(Tensor qa, Tensor b) -> (Tensor qc)'''
        pass


class QuantizedQuantizedGruCellDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::quantized_gru_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> (Tensor)'''
        pass


class QuantizedAddReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point) -> (Tensor qc)
           quantized::add_relu.Scalar(Tensor qa, Scalar b) -> (Tensor qc)
           quantized::add_relu.Scalar2(Scalar b, Tensor qa) -> (Tensor qc)'''
        pass


class QuantizedLinearReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''quantized::linear_relu(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> (Tensor Y)'''
        pass
