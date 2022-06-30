from abc import abstractmethod

from ...operators.torch.base import OperatorConverter


class ATenAbsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::abs(Tensor self) -> (Tensor)'''
        pass


class ATenAbsoluteSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::absolute(Tensor self) -> (Tensor)'''
        pass


class ATenAcosSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::acos(Tensor self) -> (Tensor)'''
        pass


class ATenAcoshSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::acosh(Tensor self) -> (Tensor)'''
        pass


class ATenAdaptiveAvgPool1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> (Tensor)'''
        pass


class ATenAdaptiveAvgPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> (Tensor)
        aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> (Tensor)'''
        pass


class ATenAdaptiveAvgPool3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_adaptive_avg_pool3d(Tensor self, int[3] output_size) -> (Tensor)
        aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> (Tensor)'''
        pass


class ATenAdaptiveMaxPool1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)'''
        pass


class ATenAdaptiveMaxPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
        aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))'''
        pass


class ATenAdaptiveMaxPool3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
        aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))'''
        pass


class ATenAddSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
        aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenAddBatchDimSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> (Tensor)'''
        pass


class ATenAddReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
        aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenAddbmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenAddcdivSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> (Tensor)'''
        pass


class ATenAddcmulSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> (Tensor)'''
        pass


class ATenAddmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenAddmmActivationSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> (Tensor)'''
        pass


class ATenAddmvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenAddrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenAdjointSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::adjoint(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenAffineGridGeneratorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> (Tensor)'''
        pass


class ATenAliasSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::alias(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenAliasCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::alias_copy(Tensor self) -> (Tensor)'''
        pass


class ATenAlignAsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::align_as(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenAlignTensorsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::align_tensors(Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenAlignToSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::align_to(Tensor(a) self, str[] names) -> (Tensor(a))
        aten::align_to.ellipsis_idx(Tensor(a) self, str[] order, int ellipsis_idx) -> (Tensor(a))'''
        pass


class ATenAllSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::all(Tensor self) -> (Tensor)
        aten::all.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor)
        aten::all.dimname(Tensor self, str dim, bool keepdim=False) -> (Tensor)'''
        pass


class ATenAlphaDropoutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::alpha_dropout(Tensor input, float p, bool train) -> (Tensor)'''
        pass


class ATenAmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> (Tensor)'''
        pass


class ATenAminSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> (Tensor)'''
        pass


class ATenAminmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_aminmax(Tensor self) -> (Tensor, Tensor)
        aten::_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
        aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
        aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)'''
        pass


class ATenAmpForeachNonFiniteCheckAndUnscaleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_amp_foreach_non_finite_check_and_unscale.functional(Tensor[] self, Tensor found_inf, Tensor inv_scale) -> (Tensor[] self_out, Tensor found_inf_out)'''
        pass


class ATenAmpUpdateScaleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_amp_update_scale.functional(Tensor self, Tensor growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> (Tensor, Tensor growth_tracker_out)'''
        pass


class ATenAndSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__and__.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::__and__.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenAngleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::angle(Tensor self) -> (Tensor)'''
        pass


class ATenAnySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::any(Tensor self) -> (Tensor)
        aten::any.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor)
        aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> (Tensor)'''
        pass


class ATenArccosSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arccos(Tensor self) -> (Tensor)'''
        pass


class ATenArccoshSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arccosh(Tensor self) -> (Tensor)'''
        pass


class ATenArcsinSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arcsin(Tensor self) -> (Tensor)'''
        pass


class ATenArcsinhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arcsinh(Tensor self) -> (Tensor)'''
        pass


class ATenArctanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arctan(Tensor self) -> (Tensor)'''
        pass


class ATenArctan2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arctan2(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenArctanhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::arctanh(Tensor self) -> (Tensor)'''
        pass


class ATenArgmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)'''
        pass


class ATenArgminSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)'''
        pass


class ATenArgsortSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::argsort(Tensor self, int dim=-1, bool descending=False) -> (Tensor)
        aten::argsort.dimname(Tensor self, str dim, bool descending=False) -> (Tensor)'''
        pass


class ATenArgwhereSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::argwhere(Tensor self) -> (Tensor)'''
        pass


class ATenAsStridedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> (Tensor(a))'''
        pass


class ATenAsStridedCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::as_strided_copy(Tensor self, int[] size, int[] stride, int? storage_offset=None) -> (Tensor)'''
        pass


class ATenAsTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::as_tensor(Tensor(a) data, *, int? dtype=None, Device? device=None) -> (Tensor(b|a))'''
        pass


class ATenAsinSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::asin(Tensor self) -> (Tensor)'''
        pass


class ATenAsinhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::asinh(Tensor self) -> (Tensor)'''
        pass


class ATenAtanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::atan(Tensor self) -> (Tensor)'''
        pass


class ATenAtan2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::atan2(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenAtanhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::atanh(Tensor self) -> (Tensor)'''
        pass


class ATenAtleast1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::atleast_1d(Tensor self) -> (Tensor)
        aten::atleast_1d.Sequence(Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenAtleast2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::atleast_2d(Tensor self) -> (Tensor)
        aten::atleast_2d.Sequence(Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenAtleast3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::atleast_3d(Tensor self) -> (Tensor)
        aten::atleast_3d.Sequence(Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenAutocastToFullPrecisionSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> (Tensor(a))'''
        pass


class ATenAutocastToReducedPrecisionSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, int cuda_dtype, int cpu_dtype) -> (Tensor(a))'''
        pass


class ATenAvgPool1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], bool ceil_mode=False, bool count_include_pad=True) -> (Tensor)'''
        pass


class ATenAvgPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> (Tensor)'''
        pass


class ATenAvgPool3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=[0, 0, 0], bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> (Tensor)'''
        pass


class ATenBaddbmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenBatchNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor)'''
        pass


class ATenBatchNormBackwardElemtSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> (Tensor)'''
        pass


class ATenBatchNormBackwardReduceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)'''
        pass


class ATenBatchNormElemtSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> (Tensor)'''
        pass


class ATenBatchNormGatherStatsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)'''
        pass


class ATenBatchNormGatherStatsWithCountsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)'''
        pass


class ATenBatchNormImplIndexSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)'''
        pass


class ATenBatchNormStatsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)'''
        pass


class ATenBatchNormUpdateStatsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)'''
        pass


class ATenBernoulliSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bernoulli(Tensor self, *, Generator? generator=None) -> (Tensor)
        aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> (Tensor)
        aten::bernoulli.Tensor_functional(Tensor self, Tensor p, *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenBilinearSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> (Tensor)'''
        pass


class ATenBinaryCrossEntropySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> (Tensor)'''
        pass


class ATenBinaryCrossEntropyWithLogitsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=1) -> (Tensor)'''
        pass


class ATenBincountSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> (Tensor)'''
        pass


class ATenBinomialSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> (Tensor)'''
        pass


class ATenBitwiseAndSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bitwise_and.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::bitwise_and.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)'''
        pass


class ATenBitwiseLeftShiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)'''
        pass


class ATenBitwiseNotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bitwise_not(Tensor self) -> (Tensor)'''
        pass


class ATenBitwiseOrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bitwise_or.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)
        aten::bitwise_or.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenBitwiseRightShiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)'''
        pass


class ATenBitwiseXorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)
        aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenBlockDiagSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::block_diag(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenBmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bmm(Tensor self, Tensor mat2) -> (Tensor)'''
        pass


class ATenBroadcastTensorsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::broadcast_tensors(Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenBroadcastToSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::broadcast_to(Tensor(a) self, int[] size) -> (Tensor(a))'''
        pass


class ATenBucketizeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> (Tensor)
        aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> (Tensor)'''
        pass


class ATenCartesianProdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cartesian_prod(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenCatSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cat(Tensor[] tensors, int dim=0) -> (Tensor)
        aten::cat.names(Tensor[] tensors, str dim) -> (Tensor)'''
        pass


class ATenCauchySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cauchy.functional(Tensor self, float median=0., float sigma=1., *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenCcolIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ccol_indices(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenCcolIndicesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ccol_indices_copy(Tensor self) -> (Tensor)'''
        pass


class ATenCdistSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cdist(Tensor x1, Tensor x2, float p=2., int? compute_mode=None) -> (Tensor)'''
        pass


class ATenCeilSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ceil(Tensor self) -> (Tensor)'''
        pass


class ATenCeluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::celu(Tensor self, Scalar alpha=1.) -> (Tensor)'''
        pass


class ATenChainMatmulSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::chain_matmul(Tensor[] matrices) -> (Tensor)'''
        pass


class ATenChalfSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::chalf(Tensor self, *, int? memory_format=None) -> (Tensor)'''
        pass


class ATenChannelShuffleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::channel_shuffle(Tensor self, int groups) -> (Tensor)'''
        pass


class ATenCholeskySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cholesky(Tensor self, bool upper=False) -> (Tensor)'''
        pass


class ATenCholeskyInverseSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cholesky_inverse(Tensor self, bool upper=False) -> (Tensor)'''
        pass


class ATenCholeskySolveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> (Tensor)'''
        pass


class ATenCholeskySolveHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> (Tensor)'''
        pass


class ATenChooseQparamsOptimizedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)'''
        pass


class ATenChunkSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::chunk(Tensor(a -> *) self, int chunks, int dim=0) -> (Tensor[])'''
        pass


class ATenClampSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> (Tensor)
        aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> (Tensor)'''
        pass


class ATenClampMaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::clamp_max(Tensor self, Scalar max) -> (Tensor)
        aten::clamp_max.Tensor(Tensor self, Tensor max) -> (Tensor)'''
        pass


class ATenClampMinSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::clamp_min(Tensor self, Scalar min) -> (Tensor)
        aten::clamp_min.Tensor(Tensor self, Tensor min) -> (Tensor)'''
        pass


class ATenClipSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> (Tensor)
        aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> (Tensor)'''
        pass


class ATenCloneSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::clone(Tensor self, *, int? memory_format=None) -> (Tensor)'''
        pass


class ATenCoalesceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_coalesce(Tensor self) -> (Tensor)
        aten::coalesce(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenCoalescedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_coalesced.functional(Tensor self, bool coalesced) -> (Tensor)'''
        pass


class ATenCol2imSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> (Tensor)'''
        pass


class ATenColIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::col_indices(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenColIndicesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::col_indices_copy(Tensor self) -> (Tensor)'''
        pass


class ATenColumnStackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::column_stack(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenCombinationsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> (Tensor)'''
        pass


class ATenComplexSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::complex(Tensor real, Tensor imag) -> (Tensor)'''
        pass


class ATenComputeLinearCombinationSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_compute_linear_combination(Tensor input, Tensor coefficients) -> (Tensor)'''
        pass


class ATenConcatSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::concat(Tensor[] tensors, int dim=0) -> (Tensor)
        aten::concat.names(Tensor[] tensors, str dim) -> (Tensor)'''
        pass


class ATenConjSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conj(Tensor(a) self) -> (Tensor(a))
        aten::_conj(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenConjCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_conj_copy(Tensor self) -> (Tensor)'''
        pass


class ATenConjPhysicalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conj_physical(Tensor self) -> (Tensor)
        aten::_conj_physical(Tensor self) -> (Tensor)'''
        pass


class ATenConstantPadNdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> (Tensor)'''
        pass


class ATenContiguousSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::contiguous(Tensor(a) self, *, int memory_format=0) -> (Tensor(a))'''
        pass


class ATenConv1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=[1], int[1] padding=[0], int[1] dilation=[1], int groups=1) -> (Tensor)
        aten::conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=[1], str padding="valid", int[1] dilation=[1], int groups=1) -> (Tensor)'''
        pass


class ATenConv2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=[1, 1], int[2] padding=[0, 0], int[2] dilation=[1, 1], int groups=1) -> (Tensor)
        aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=[1, 1], str padding="valid", int[2] dilation=[1, 1], int groups=1) -> (Tensor)'''
        pass


class ATenConv3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=[1, 1, 1], int[3] padding=[0, 0, 0], int[3] dilation=[1, 1, 1], int groups=1) -> (Tensor)
        aten::conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=[1, 1, 1], str padding="valid", int[3] dilation=[1, 1, 1], int groups=1) -> (Tensor)'''
        pass


class ATenConvDepthwise2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> (Tensor)'''
        pass


class ATenConvDepthwise3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> (Tensor)'''
        pass


class ATenConvTbcSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> (Tensor)'''
        pass


class ATenConvTranspose1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=[1], int[1] padding=[0], int[1] output_padding=[0], int groups=1, int[1] dilation=[1]) -> (Tensor)'''
        pass


class ATenConvTranspose2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=[1, 1], int[2] padding=[0, 0], int[2] output_padding=[0, 0], int groups=1, int[2] dilation=[1, 1]) -> (Tensor)'''
        pass


class ATenConvTranspose3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=[1, 1, 1], int[3] padding=[0, 0, 0], int[3] output_padding=[0, 0, 0], int groups=1, int[3] dilation=[1, 1, 1]) -> (Tensor)'''
        pass


class ATenConvertIndicesFromCooToCsrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> (Tensor)'''
        pass


class ATenConvertIndicesFromCsrToCooSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_convert_indices_from_csr_to_coo(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False) -> (Tensor)'''
        pass


class ATenConvolutionSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> (Tensor)
        aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> (Tensor)
        aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> (Tensor)'''
        pass


class ATenConvolutionBackwardOverrideableSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)'''
        pass


class ATenConvolutionModeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[] dilation, int groups) -> (Tensor)'''
        pass


class ATenConvolutionOverrideableSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> (Tensor)'''
        pass


class ATenCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::copy(Tensor self, Tensor src, bool non_blocking=False) -> (Tensor)
        aten::copy.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> (Tensor(a!))
        aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> (Tensor(a!))
        aten::copy_.Tensor(Tensor(a!) self, Tensor other) -> (Tensor(a!))
        aten::copy_.int(Tensor(a!) self, int other) -> (Tensor(a!))
        aten::copy_.float(Tensor(a!) self, float other) -> (Tensor(a!))'''
        pass


class ATenCopyFromSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> (Tensor)'''
        pass


class ATenCopyFromAndResizeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_copy_from_and_resize(Tensor self, Tensor dst) -> (Tensor)'''
        pass


class ATenCopysignSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::copysign.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::copysign.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenCorrcoefSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::corrcoef(Tensor self) -> (Tensor)'''
        pass


class ATenCosSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cos(Tensor self) -> (Tensor)'''
        pass


class ATenCoshSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cosh(Tensor self) -> (Tensor)'''
        pass


class ATenCosineEmbeddingLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0., int reduction=1) -> (Tensor)'''
        pass


class ATenCosineSimilaritySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> (Tensor)'''
        pass


class ATenCountNonzeroSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> (Tensor)
        aten::count_nonzero(Tensor self, int? dim=None) -> (Tensor)'''
        pass


class ATenCovSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> (Tensor)'''
        pass


class ATenCpuSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cpu(Tensor(a) self) -> (Tensor(b|a))'''
        pass


class ATenCrossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cross(Tensor self, Tensor other, int? dim=None) -> (Tensor)'''
        pass


class ATenCrossEntropyLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, int ignore_index=-100, float label_smoothing=0.) -> (Tensor)'''
        pass


class ATenCrowIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::crow_indices(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenCrowIndicesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::crow_indices_copy(Tensor self) -> (Tensor)'''
        pass


class ATenCtcLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=1, bool zero_infinity=False) -> (Tensor)
        aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=1, bool zero_infinity=False) -> (Tensor)
        aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)'''
        pass


class ATenCummaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
        aten::cummax.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)
        aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenCumminSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
        aten::cummin.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)
        aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenCumprodSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cumprod(Tensor self, int dim, *, int? dtype=None) -> (Tensor)
        aten::cumprod.dimname(Tensor self, str dim, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenCumsumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cumsum(Tensor self, int dim, *, int? dtype=None) -> (Tensor)
        aten::cumsum.dimname(Tensor self, str dim, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenCumulativeTrapezoidSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> (Tensor)
        aten::cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> (Tensor)'''
        pass


class ATenDataSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::data(Tensor self) -> (Tensor)'''
        pass


class ATenDeg2radSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::deg2rad(Tensor self) -> (Tensor)'''
        pass


class ATenDequantizeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::dequantize.self(Tensor self) -> (Tensor)
        aten::dequantize.tensors(Tensor[] tensors) -> (Tensor[])
        aten::dequantize.tensor(Tensor qtensor) -> (Tensor)
        aten::dequantize.list(Tensor[] qtensors) -> (Tensor[])'''
        pass


class ATenDetSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::det(Tensor self) -> (Tensor)'''
        pass


class ATenDetLuBasedHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_det_lu_based_helper(Tensor self) -> (Tensor det, Tensor lu, Tensor pivs)'''
        pass


class ATenDetLuBasedHelperBackwardHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_det_lu_based_helper_backward_helper(Tensor det_grad, Tensor det, Tensor self, Tensor lu, Tensor pivs) -> (Tensor)'''
        pass


class ATenDetachSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::detach(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenDetachCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::detach_copy(Tensor self) -> (Tensor)'''
        pass


class ATenDiagSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diag(Tensor self, int diagonal=0) -> (Tensor)'''
        pass


class ATenDiagEmbedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> (Tensor)'''
        pass


class ATenDiagflatSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diagflat(Tensor self, int offset=0) -> (Tensor)'''
        pass


class ATenDiagonalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> (Tensor(a))
        aten::diagonal.Dimname(Tensor(a) self, *, str outdim, str dim1, str dim2, int offset=0) -> (Tensor(a))'''
        pass


class ATenDiagonalCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> (Tensor)'''
        pass


class ATenDiagonalScatterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> (Tensor)'''
        pass


class ATenDiffSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> (Tensor)'''
        pass


class ATenDigammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::digamma(Tensor self) -> (Tensor)'''
        pass


class ATenDimArangeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_dim_arange(Tensor like, int dim) -> (Tensor)'''
        pass


class ATenDistSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::dist(Tensor self, Tensor other, Scalar p=2) -> (Tensor)'''
        pass


class ATenDivSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::div.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> (Tensor)
        aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> (Tensor)'''
        pass


class ATenDivideSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::divide.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::divide.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> (Tensor)
        aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> (Tensor)'''
        pass


class ATenDotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::dot(Tensor self, Tensor tensor) -> (Tensor)'''
        pass


class ATenDropoutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::dropout(Tensor input, float p, bool train) -> (Tensor)'''
        pass


class ATenDsplitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::dsplit.int(Tensor(a -> *) self, int sections) -> (Tensor[])
        aten::dsplit.array(Tensor(a -> *) self, int[] indices) -> (Tensor[])'''
        pass


class ATenDstackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::dstack(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenEigSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
        aten::eig.e(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)'''
        pass


class ATenEinsumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::einsum(str equation, Tensor[] tensors) -> (Tensor)
        aten::einsum.sublist(Tensor a, ...) -> (Tensor)'''
        pass


class ATenEluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)'''
        pass


class ATenEmbeddingSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> (Tensor)'''
        pass


class ATenEmbeddingBagSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
        aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)
        aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)'''
        pass


class ATenEmbeddingRenormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::embedding_renorm.functional(Tensor self, Tensor indices, float max_norm, float norm_type) -> (Tensor)'''
        pass


class ATenEmptyLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::empty_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenEmptyPerChannelAffineQuantizedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=0) -> (Tensor)'''
        pass


class ATenEmptyQuantizedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::empty_quantized(int[] size, Tensor qtensor, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenEqSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::eq.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::eq.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenErfSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::erf(Tensor self) -> (Tensor)'''
        pass


class ATenErfcSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::erfc(Tensor self) -> (Tensor)'''
        pass


class ATenErfinvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::erfinv(Tensor self) -> (Tensor)'''
        pass


class ATenEuclideanDistSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_euclidean_dist(Tensor x1, Tensor x2) -> (Tensor)'''
        pass


class ATenExpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::exp(Tensor self) -> (Tensor)'''
        pass


class ATenExp2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::exp2(Tensor self) -> (Tensor)'''
        pass


class ATenExpandSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> (Tensor(a))
        aten::expand.SymInt(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> (Tensor(a))'''
        pass


class ATenExpandAsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::expand_as(Tensor(a) self, Tensor other) -> (Tensor(a))'''
        pass


class ATenExpandCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::expand_copy(Tensor self, int[] size, *, bool implicit=False) -> (Tensor)
        aten::expand_copy.SymInt(Tensor self, SymInt[] size, *, bool implicit=False) -> (Tensor)'''
        pass


class ATenExpm1Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::expm1(Tensor self) -> (Tensor)'''
        pass


class ATenExponentialSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::exponential.functional(Tensor self, float lambd=1., *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenFakeQuantizeLearnablePerChannelAffineSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.) -> (Tensor)'''
        pass


class ATenFakeQuantizeLearnablePerTensorAffineSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.) -> (Tensor)'''
        pass


class ATenFakeQuantizePerChannelAffineSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor)'''
        pass


class ATenFakeQuantizePerChannelAffineCachemaskSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)'''
        pass


class ATenFakeQuantizePerTensorAffineSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor)
        aten::fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> (Tensor)'''
        pass


class ATenFakeQuantizePerTensorAffineCachemaskSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)'''
        pass


class ATenFakeQuantizePerTensorAffineCachemaskTensorQparamsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)'''
        pass


class ATenFeatureAlphaDropoutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::feature_alpha_dropout(Tensor input, float p, bool train) -> (Tensor)'''
        pass


class ATenFeatureDropoutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::feature_dropout(Tensor input, float p, bool train) -> (Tensor)'''
        pass


class ATenFftC2cSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fft_c2c(Tensor self, int[] dim, int normalization, bool forward) -> (Tensor)'''
        pass


class ATenFftC2rSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fft_c2r(Tensor self, int[] dim, int normalization, int last_dim_size) -> (Tensor)'''
        pass


class ATenFftFftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> (Tensor)'''
        pass


class ATenFftFft2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> (Tensor)'''
        pass


class ATenFftFftnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> (Tensor)'''
        pass


class ATenFftFftshiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_fftshift(Tensor self, int[1]? dim=None) -> (Tensor)'''
        pass


class ATenFftHfftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> (Tensor)'''
        pass


class ATenFftHfft2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_hfft2(Tensor self, int[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> (Tensor)'''
        pass


class ATenFftHfftnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_hfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> (Tensor)'''
        pass


class ATenFftIfftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> (Tensor)'''
        pass


class ATenFftIfft2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> (Tensor)'''
        pass


class ATenFftIfftnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> (Tensor)'''
        pass


class ATenFftIfftshiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> (Tensor)'''
        pass


class ATenFftIhfftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> (Tensor)'''
        pass


class ATenFftIhfft2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ihfft2(Tensor self, int[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> (Tensor)'''
        pass


class ATenFftIhfftnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_ihfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> (Tensor)'''
        pass


class ATenFftIrfftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> (Tensor)'''
        pass


class ATenFftIrfft2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> (Tensor)'''
        pass


class ATenFftIrfftnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> (Tensor)'''
        pass


class ATenFftR2cSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> (Tensor)'''
        pass


class ATenFftRfftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> (Tensor)'''
        pass


class ATenFftRfft2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> (Tensor)'''
        pass


class ATenFftRfftnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> (Tensor)'''
        pass


class ATenFillSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fill.Scalar(Tensor self, Scalar value) -> (Tensor)
        aten::fill.Tensor(Tensor self, Tensor value) -> (Tensor)
        aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> (Tensor(a!))
        aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> (Tensor(a!))'''
        pass


class ATenFixSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fix(Tensor self) -> (Tensor)'''
        pass


class ATenFlattenSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::flatten.DimnameList(Tensor(a) self, str[] dims, str out_dim) -> (Tensor(a))
        aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, str out_dim) -> (Tensor(a))
        aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> (Tensor(a))
        aten::flatten.using_names(Tensor(a) self, str start_dim, str end_dim, str out_dim) -> (Tensor(a))'''
        pass


class ATenFlattenDenseTensorsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::flatten_dense_tensors(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenFlipSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::flip(Tensor self, int[] dims) -> (Tensor)'''
        pass


class ATenFliplrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fliplr(Tensor self) -> (Tensor)'''
        pass


class ATenFlipudSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::flipud(Tensor self) -> (Tensor)'''
        pass


class ATenFloatPowerSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)
        aten::float_power.Scalar(Scalar self, Tensor exponent) -> (Tensor)
        aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> (Tensor)'''
        pass


class ATenFloorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::floor(Tensor self) -> (Tensor)'''
        pass


class ATenFloorDivideSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::floor_divide(Tensor self, Tensor other) -> (Tensor)
        aten::floor_divide.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenFmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fmax(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenFminSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fmin(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenFmodSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fmod.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::fmod.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenFracSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::frac(Tensor self) -> (Tensor)'''
        pass


class ATenFractionalMaxPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
        aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))'''
        pass


class ATenFractionalMaxPool3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
        aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))'''
        pass


class ATenFrexpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)'''
        pass


class ATenFrobeniusNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::frobenius_norm(Tensor self) -> (Tensor)
        aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> (Tensor)'''
        pass


class ATenFullLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenFusedDropoutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)'''
        pass


class ATenFusedMovingAvgObsFakeQuantSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor)'''
        pass


class ATenFusedMovingAvgObsFqHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
        aten::_fused_moving_avg_obs_fq_helper.functional(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask, Tensor running_min_out, Tensor running_max_out, Tensor scale_out, Tensor zero_point_out)
        aten::_fused_moving_avg_obs_fq_helper.out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))'''
        pass


class ATenFwPrimalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fw_primal(Tensor(a) self, int level) -> (Tensor(a))'''
        pass


class ATenFwPrimalCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_fw_primal_copy(Tensor self, int level) -> (Tensor)'''
        pass


class ATenGatherSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> (Tensor)
        aten::gather.dimname(Tensor self, str dim, Tensor index, *, bool sparse_grad=False) -> (Tensor)'''
        pass


class ATenGcdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::gcd(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenGeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ge.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::ge.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenGeluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::gelu(Tensor self, *, str approximate="none") -> (Tensor)'''
        pass


class ATenGeometricSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::geometric.functional(Tensor self, float p, *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenGeqrfSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
        aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)'''
        pass


class ATenGerSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ger(Tensor self, Tensor vec2) -> (Tensor)'''
        pass


class ATenGluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::glu(Tensor self, int dim=-1) -> (Tensor)'''
        pass


class ATenGluBackwardJvpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::glu_backward_jvp(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim) -> (Tensor)'''
        pass


class ATenGluJvpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::glu_jvp(Tensor glu, Tensor x, Tensor dx, int dim) -> (Tensor)'''
        pass


class ATenGreaterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::greater.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::greater.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenGreaterEqualSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::greater_equal.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::greater_equal.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenGridSamplerSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor)'''
        pass


class ATenGridSampler2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor)'''
        pass


class ATenGridSampler2dCpuFallbackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor)'''
        pass


class ATenGridSampler3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor)'''
        pass


class ATenGroupNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1.0000000000000001e-05, bool cudnn_enabled=True) -> (Tensor)'''
        pass


class ATenGruSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
        aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)'''
        pass


class ATenGruCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor)'''
        pass


class ATenGtSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::gt.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::gt.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenHardshrinkSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hardshrink(Tensor self, Scalar lambd=0.5) -> (Tensor)'''
        pass


class ATenHardsigmoidSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hardsigmoid(Tensor self) -> (Tensor)'''
        pass


class ATenHardswishSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hardswish(Tensor self) -> (Tensor)'''
        pass


class ATenHardtanhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor)'''
        pass


class ATenHeavisideSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::heaviside(Tensor self, Tensor values) -> (Tensor)'''
        pass


class ATenHingeEmbeddingLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1., int reduction=1) -> (Tensor)'''
        pass


class ATenHistcSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> (Tensor)'''
        pass


class ATenHistogramSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
        aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)'''
        pass


class ATenHistogramddSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::histogramdd(Tensor self, int[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
        aten::histogramdd.int_bins(Tensor self, int bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
        aten::histogramdd.TensorList_bins(Tensor self, Tensor[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)'''
        pass


class ATenHistogramddBinEdgesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_histogramdd_bin_edges(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor[])'''
        pass


class ATenHistogramddFromBinCtsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_histogramdd_from_bin_cts(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor)'''
        pass


class ATenHistogramddFromBinTensorsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_histogramdd_from_bin_tensors(Tensor self, Tensor[] bins, *, Tensor? weight=None, bool density=False) -> (Tensor)'''
        pass


class ATenHsplitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hsplit.int(Tensor(a -> *) self, int sections) -> (Tensor[])
        aten::hsplit.array(Tensor(a -> *) self, int[] indices) -> (Tensor[])'''
        pass


class ATenHspmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hspmm(Tensor mat1, Tensor mat2) -> (Tensor)'''
        pass


class ATenHstackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hstack(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenHuberLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::huber_loss(Tensor self, Tensor target, int reduction=1, float delta=1.) -> (Tensor)'''
        pass


class ATenHypotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::hypot(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenI0Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::i0(Tensor self) -> (Tensor)'''
        pass


class ATenIgammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::igamma(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenIgammacSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::igammac(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenIm2colSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> (Tensor)'''
        pass


class ATenImagSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::imag(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenIndexSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index.Tensor(Tensor self, Tensor?[] indices) -> (Tensor)
        aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> (Tensor)'''
        pass


class ATenIndexAddSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> (Tensor)
        aten::index_add.dimname(Tensor self, str dim, Tensor index, Tensor source, *, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenIndexCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> (Tensor)
        aten::index_copy.dimname(Tensor self, str dim, Tensor index, Tensor source) -> (Tensor)'''
        pass


class ATenIndexFillSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index_fill.Dimname_Scalar(Tensor self, str dim, Tensor index, Scalar value) -> (Tensor)
        aten::index_fill.Dimname_Tensor(Tensor self, str dim, Tensor index, Tensor value) -> (Tensor)
        aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
        aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> (Tensor)'''
        pass


class ATenIndexPutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> (Tensor)
        aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> (Tensor)'''
        pass


class ATenIndexPutImplSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_index_put_impl.functional(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> (Tensor)'''
        pass


class ATenIndexReduceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> (Tensor)'''
        pass


class ATenIndexSelectSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::index_select(Tensor self, int dim, Tensor index) -> (Tensor)
        aten::index_select.dimname(Tensor self, str dim, Tensor index) -> (Tensor)'''
        pass


class ATenIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::indices(Tensor(a) self) -> (Tensor(a))
        aten::_indices(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenIndicesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::indices_copy(Tensor self) -> (Tensor)
        aten::_indices_copy(Tensor self) -> (Tensor)'''
        pass


class ATenInnerSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::inner(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenInstanceNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> (Tensor)'''
        pass


class ATenIntReprSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::int_repr(Tensor self) -> (Tensor)'''
        pass


class ATenInterpolateSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__interpolate.scale_list(Tensor input, int? size=None, float[]? scale_factor=None, str mode="nearest", bool? align_corners=None, bool? recompute_scale_factor=None, bool antialias=False) -> (Tensor)
        aten::__interpolate.size_list_scale_list(Tensor input, int[]? size=None, float[]? scale_factor=None, str mode="nearest", bool? align_corners=None, bool? recompute_scale_factor=None, bool antialias=False) -> (Tensor)
        aten::__interpolate(Tensor input, int? size=None, float? scale_factor=None, str mode="nearest", bool? align_corners=None, bool? recompute_scale_factor=None, bool antialias=False) -> (Tensor)
        aten::__interpolate.size_list(Tensor input, int[]? size=None, float? scale_factor=None, str mode="nearest", bool? align_corners=None, bool? recompute_scale_factor=None, bool antialias=False) -> (Tensor)'''
        pass


class ATenInverseSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::inverse(Tensor self) -> (Tensor)'''
        pass


class ATenIstftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> (Tensor)'''
        pass


class ATenKeysSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::keys.Tensor(Dict(Tensor, t) self) -> (Tensor[](*))'''
        pass


class ATenKlDivSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::kl_div(Tensor self, Tensor target, int reduction=1, *, bool log_target=False) -> (Tensor)'''
        pass


class ATenKronSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::kron(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenKthvalueSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::kthvalue.dimname(Tensor self, int k, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenL1LossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::l1_loss(Tensor self, Tensor target, int reduction=1) -> (Tensor)'''
        pass


class ATenLayerNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1.0000000000000001e-05, bool cudnn_enable=True) -> (Tensor)'''
        pass


class ATenLcmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lcm(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLdexpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ldexp.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::le.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::le.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenLeakyReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> (Tensor)'''
        pass


class ATenLerpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> (Tensor)
        aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> (Tensor)'''
        pass


class ATenLessSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::less.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::less.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLessEqualSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::less_equal.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::less_equal.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLgammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lgamma(Tensor self) -> (Tensor)'''
        pass


class ATenLiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lift(Tensor self) -> (Tensor)'''
        pass


class ATenLinalgCholeskySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_cholesky(Tensor self, *, bool upper=False) -> (Tensor)'''
        pass


class ATenLinalgCholeskyExSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
        aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)'''
        pass


class ATenLinalgCondSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_cond(Tensor self, Scalar? p=None) -> (Tensor)
        aten::linalg_cond.p_str(Tensor self, str p) -> (Tensor)'''
        pass


class ATenLinalgCrossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> (Tensor)'''
        pass


class ATenLinalgDetSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_det(Tensor self) -> (Tensor)'''
        pass


class ATenLinalgDiagonalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> (Tensor(a))'''
        pass


class ATenLinalgEigSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
        aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)'''
        pass


class ATenLinalgEighSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
        aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)'''
        pass


class ATenLinalgEigvalsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_eigvals(Tensor self) -> (Tensor)'''
        pass


class ATenLinalgEigvalshSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> (Tensor)'''
        pass


class ATenLinalgHouseholderProductSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_householder_product(Tensor input, Tensor tau) -> (Tensor)'''
        pass


class ATenLinalgInvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_inv(Tensor self) -> (Tensor)'''
        pass


class ATenLinalgInvExSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_inv_ex(Tensor self, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
        aten::linalg_inv_ex.inverse(Tensor self, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)'''
        pass


class ATenLinalgInvOutHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_linalg_inv_out_helper.functional(Tensor self, Tensor infos_lu, Tensor infos_getri) -> (Tensor, Tensor infos_lu_out, Tensor infos_getri_out)'''
        pass


class ATenLinalgLdlFactorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_ldl_factor(Tensor self, *, bool hermitian=False) -> (Tensor LD, Tensor pivots)
        aten::linalg_ldl_factor.out(Tensor self, *, bool hermitian=False, Tensor(a!) LD, Tensor(b!) pivots) -> (Tensor(a!) LD, Tensor(b!) pivots)'''
        pass


class ATenLinalgLdlFactorExSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)
        aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info)'''
        pass


class ATenLinalgLdlSolveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> (Tensor)'''
        pass


class ATenLinalgLstsqSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
        aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)'''
        pass


class ATenLinalgLuSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)
        aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)'''
        pass


class ATenLinalgLuFactorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_lu_factor(Tensor A, *, bool pivot=True) -> (Tensor LU, Tensor pivots)
        aten::linalg_lu_factor.out(Tensor A, *, bool pivot=True, Tensor(a!) LU, Tensor(b!) pivots) -> (Tensor(a!) LU, Tensor(b!) pivots)'''
        pass


class ATenLinalgLuFactorExSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)
        aten::linalg_lu_factor_ex.out(Tensor A, *, bool pivot=True, bool check_errors=False, Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info)'''
        pass


class ATenLinalgMatmulSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_matmul(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLinalgMatrixExpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_matrix_exp(Tensor self) -> (Tensor)'''
        pass


class ATenLinalgMatrixNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2, -1], bool keepdim=False, *, int? dtype=None) -> (Tensor)
        aten::linalg_matrix_norm.str_ord(Tensor self, str ord="fro", int[] dim=[-2, -1], bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenLinalgMatrixPowerSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_matrix_power(Tensor self, int n) -> (Tensor)'''
        pass


class ATenLinalgMatrixRankSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_matrix_rank(Tensor self, float tol, bool hermitian=False) -> (Tensor)
        aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> (Tensor)
        aten::linalg_matrix_rank.atol_rtol_tensor(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> (Tensor)
        aten::linalg_matrix_rank.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> (Tensor)'''
        pass


class ATenLinalgMultiDotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_multi_dot(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenLinalgNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, int? dtype=None) -> (Tensor)
        aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenLinalgPinvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_pinv.atol_rtol_tensor(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> (Tensor)
        aten::linalg_pinv.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> (Tensor)
        aten::linalg_pinv(Tensor self, float rcond, bool hermitian=False) -> (Tensor)
        aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> (Tensor)'''
        pass


class ATenLinalgQrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)
        aten::linalg_qr.out(Tensor A, str mode="reduced", *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)'''
        pass


class ATenLinalgQrHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_linalg_qr_helper(Tensor self, str mode) -> (Tensor, Tensor)'''
        pass


class ATenLinalgSlogdetSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
        aten::linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)'''
        pass


class ATenLinalgSolveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_solve(Tensor input, Tensor other) -> (Tensor)'''
        pass


class ATenLinalgSolveTriangularSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> (Tensor)'''
        pass


class ATenLinalgSvdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor Vh)
        aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
        aten::linalg_svd(Tensor A, bool full_matrices=True) -> (Tensor U, Tensor S, Tensor Vh)
        aten::linalg_svd.U(Tensor A, bool full_matrices=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)'''
        pass


class ATenLinalgSvdvalsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_svdvals(Tensor A) -> (Tensor)'''
        pass


class ATenLinalgTensorinvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_tensorinv(Tensor self, int ind=2) -> (Tensor)'''
        pass


class ATenLinalgTensorsolveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> (Tensor)'''
        pass


class ATenLinalgVanderSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_vander(Tensor x, *, int? N=None) -> (Tensor)'''
        pass


class ATenLinalgVectorNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenLinearSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> (Tensor)'''
        pass


class ATenLogSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log(Tensor self) -> (Tensor)'''
        pass


class ATenLog10Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log10(Tensor self) -> (Tensor)'''
        pass


class ATenLog1pSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log1p(Tensor self) -> (Tensor)'''
        pass


class ATenLog2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log2(Tensor self) -> (Tensor)'''
        pass


class ATenLogNormalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log_normal.functional(Tensor self, float mean=1., float std=2., *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenLogSigmoidSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log_sigmoid(Tensor self) -> (Tensor)'''
        pass


class ATenLogSoftmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::log_softmax.int(Tensor self, int dim, int? dtype=None) -> (Tensor)
        aten::log_softmax.Dimname(Tensor self, str dim, *, int? dtype=None) -> (Tensor)
        aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> (Tensor)'''
        pass


class ATenLogSoftmaxBackwardDataSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, int input_dtype) -> (Tensor)'''
        pass


class ATenLogaddexpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logaddexp(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLogaddexp2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logaddexp2(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLogcumsumexpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logcumsumexp(Tensor self, int dim) -> (Tensor)
        aten::logcumsumexp.dimname(Tensor self, str dim) -> (Tensor)
        aten::_logcumsumexp(Tensor self, int dim) -> (Tensor)'''
        pass


class ATenLogdetSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logdet(Tensor self) -> (Tensor)'''
        pass


class ATenLogicalAndSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logical_and(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLogicalNotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logical_not(Tensor self) -> (Tensor)'''
        pass


class ATenLogicalOrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logical_or(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLogicalXorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logical_xor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLogitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logit(Tensor self, float? eps=None) -> (Tensor)'''
        pass


class ATenLogsumexpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> (Tensor)
        aten::logsumexp.names(Tensor self, str[1] dim, bool keepdim=False) -> (Tensor)'''
        pass


class ATenLshiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__lshift__.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::__lshift__.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenLstmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
        aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenLstmCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)'''
        pass


class ATenLstmMpsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_lstm_mps(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor)'''
        pass


class ATenLstsqSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
        aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)'''
        pass


class ATenLtSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lt.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::lt.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenLuSolveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> (Tensor)'''
        pass


class ATenLuWithInfoSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)'''
        pass


class ATenMHSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mH(Tensor(a) self) -> (Tensor(a))
        aten::mH.a(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenMTSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mT(Tensor(a) self) -> (Tensor(a))
        aten::mT.a(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenMakeDualSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> (Tensor(a))'''
        pass


class ATenMakeDualCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_make_dual_copy(Tensor primal, Tensor tangent, int level) -> (Tensor)'''
        pass


class ATenMakePerChannelQuantizedTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> (Tensor)'''
        pass


class ATenMakePerTensorQuantizedTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> (Tensor)'''
        pass


class ATenMarginRankingLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0., int reduction=1) -> (Tensor)'''
        pass


class ATenMaskedFillSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> (Tensor)
        aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> (Tensor)'''
        pass


class ATenMaskedScaleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_masked_scale(Tensor self, Tensor mask, float scale) -> (Tensor)'''
        pass


class ATenMaskedScatterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> (Tensor)'''
        pass


class ATenMaskedSelectSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::masked_select(Tensor self, Tensor mask) -> (Tensor)'''
        pass


class ATenMaskedSoftmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_masked_softmax(Tensor self, Tensor mask, int? dim=None) -> (Tensor)'''
        pass


class ATenMatmulSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::matmul(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenMatrixExpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::matrix_exp(Tensor self) -> (Tensor)'''
        pass


class ATenMatrixHSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::matrix_H(Tensor(a) self) -> (Tensor(a))
        aten::matrix_H.a(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenMatrixPowerSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::matrix_power(Tensor self, int n) -> (Tensor)'''
        pass


class ATenMatrixRankSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::matrix_rank(Tensor self, bool symmetric=False) -> (Tensor)
        aten::matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> (Tensor)'''
        pass


class ATenMaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max(Tensor self) -> (Tensor)
        aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::max.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::max.names_dim_max(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::max.other(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenMaxPool1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> (Tensor)'''
        pass


class ATenMaxPool1dWithIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> (Tensor, Tensor)'''
        pass


class ATenMaxPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False) -> (Tensor)'''
        pass


class ATenMaxPool2dWithIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False) -> (Tensor, Tensor)
        aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))'''
        pass


class ATenMaxPool3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=[0, 0, 0], int[3] dilation=[1, 1, 1], bool ceil_mode=False) -> (Tensor)'''
        pass


class ATenMaxPool3dWithIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=[0, 0, 0], int[3] dilation=[1, 1, 1], bool ceil_mode=False) -> (Tensor, Tensor)
        aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=[0, 0, 0], int[3] dilation=[1, 1, 1], bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))'''
        pass


class ATenMaxUnpool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> (Tensor)'''
        pass


class ATenMaxUnpool3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> (Tensor)'''
        pass


class ATenMaximumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::maximum(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenMeanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mean(Tensor self, *, int? dtype=None) -> (Tensor)
        aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)
        aten::mean.names_dim(Tensor self, str[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenMedianSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::median(Tensor self) -> (Tensor)
        aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::median.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::median.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenMeshgridSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::meshgrid(Tensor[] tensors) -> (Tensor[])
        aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> (Tensor[])'''
        pass


class ATenMinSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::min(Tensor self) -> (Tensor)
        aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::min.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::min.names_dim_min(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::min.other(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenMinimumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::minimum(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenMishSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mish(Tensor self) -> (Tensor)'''
        pass


class ATenMmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mm(Tensor self, Tensor mat2) -> (Tensor)'''
        pass


class ATenModeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::mode.dimname(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenMoveaxisSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> (Tensor(a))
        aten::moveaxis.int(Tensor(a) self, int source, int destination) -> (Tensor(a))'''
        pass


class ATenMovedimSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> (Tensor(a))
        aten::movedim.int(Tensor(a) self, int source, int destination) -> (Tensor(a))'''
        pass


class ATenMpsConvolutionSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_mps_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> (Tensor)'''
        pass


class ATenMpsConvolutionTransposeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_mps_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups) -> (Tensor)'''
        pass


class ATenMpsLinearSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_mps_linear(Tensor self, Tensor weight, Tensor? bias=None) -> (Tensor)'''
        pass


class ATenMpsLinearBackwardInputSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_mps_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> (Tensor)'''
        pass


class ATenMpsLinearBackwardWeightsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_mps_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)'''
        pass


class ATenMpsMaxPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_mps_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False) -> (Tensor)'''
        pass


class ATenMseLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mse_loss(Tensor self, Tensor target, int reduction=1) -> (Tensor)'''
        pass


class ATenMsortSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::msort(Tensor self) -> (Tensor)'''
        pass


class ATenMulSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mul.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::mul.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenMultiMarginLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=1) -> (Tensor)'''
        pass


class ATenMultilabelMarginLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=1) -> (Tensor)'''
        pass


class ATenMultinomialSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenMultiplySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::multiply.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::multiply.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenMvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mv(Tensor self, Tensor vec) -> (Tensor)'''
        pass


class ATenMvlgammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::mvlgamma(Tensor self, int p) -> (Tensor)'''
        pass


class ATenNanToNumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> (Tensor)'''
        pass


class ATenNanmeanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nanmean(Tensor self, int[1] dim=[], bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenNanmedianSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nanmedian(Tensor self) -> (Tensor)
        aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::nanmedian.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        aten::nanmedian.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenNanquantileSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> (Tensor)
        aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> (Tensor)'''
        pass


class ATenNansumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nansum(Tensor self, int[1] dim=[], bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenNarrowSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::narrow(Tensor(a) self, int dim, int start, int length) -> (Tensor(a))
        aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> (Tensor(a))'''
        pass


class ATenNarrowCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::narrow_copy(Tensor self, int dim, int start, int length) -> (Tensor)
        aten::narrow_copy.SymInt(Tensor self, int dim, int start, SymInt length) -> (Tensor)'''
        pass


class ATenNativeBatchNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))'''
        pass


class ATenNativeChannelShuffleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::native_channel_shuffle(Tensor self, int groups) -> (Tensor)'''
        pass


class ATenNativeDropoutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)'''
        pass


class ATenNativeGroupNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenNativeLayerNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenNativeMultiHeadAttentionSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_native_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True) -> (Tensor, Tensor)'''
        pass


class ATenNativeNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::native_norm(Tensor self, Scalar p=2) -> (Tensor)
        aten::native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, int? dtype) -> (Tensor)'''
        pass


class ATenNcfUnsqueezeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_ncf_unsqueeze(Tensor(a) self, int ndim) -> (Tensor(a))'''
        pass


class ATenNcfViewSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_ncf_view(Tensor(a) self, int[] input_shape, int normalized_ndim) -> (Tensor(a))'''
        pass


class ATenNeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ne.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::ne.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenNegSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::neg(Tensor self) -> (Tensor)'''
        pass


class ATenNegViewSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_neg_view(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenNegViewCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_neg_view_copy(Tensor self) -> (Tensor)'''
        pass


class ATenNegativeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::negative(Tensor self) -> (Tensor)'''
        pass


class ATenNestedFromPaddedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_nested_from_padded(Tensor padded, Tensor cpu_nested_shape_example, bool fuse_transform_0213=False) -> (Tensor)'''
        pass


class ATenNestedFromPaddedAndNestedExampleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_nested_from_padded_and_nested_example(Tensor padded, Tensor nt_example) -> (Tensor)'''
        pass


class ATenNestedTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nested_tensor(Tensor[] list, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)'''
        pass


class ATenNestedTensorFromMaskSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_nested_tensor_from_mask(Tensor t, Tensor mask) -> (Tensor)'''
        pass


class ATenNestedTensorLayerNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_nested_tensor_layer_norm(Tensor self, Tensor? weight, Tensor? bias, float eps) -> (Tensor)'''
        pass


class ATenNewEmptySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::new_empty(Tensor self, int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)'''
        pass


class ATenNewEmptyStridedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::new_empty_strided(Tensor self, int[] size, int[] stride, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)'''
        pass


class ATenNewFullSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::new_full(Tensor self, int[] size, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)'''
        pass


class ATenNewOnesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::new_ones(Tensor self, int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)'''
        pass


class ATenNewZerosSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::new_zeros(Tensor self, int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)'''
        pass


class ATenNewZerosWithSameFeatureMetaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_new_zeros_with_same_feature_meta(Tensor self, Tensor other, *, int self_num_batch_dims=0) -> (Tensor)'''
        pass


class ATenNextafterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nextafter(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenNllLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, int ignore_index=-100) -> (Tensor)'''
        pass


class ATenNllLoss2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, int ignore_index=-100) -> (Tensor)'''
        pass


class ATenNllLossNdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, int ignore_index=-100) -> (Tensor)'''
        pass


class ATenNonzeroSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nonzero(Tensor self) -> (Tensor)'''
        pass


class ATenNonzeroNumpySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nonzero_numpy(Tensor self) -> (Tensor[])'''
        pass


class ATenNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::norm.Scalar(Tensor self, Scalar p=2) -> (Tensor)
        aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> (Tensor)
        aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, str[1] dim, bool keepdim=False) -> (Tensor)
        aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, int dtype) -> (Tensor)
        aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, int dtype) -> (Tensor)
        aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, int dtype) -> (Tensor)'''
        pass


class ATenNormExceptDimSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> (Tensor)'''
        pass


class ATenNormalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::normal.Tensor_float(Tensor mean, float std=1., *, Generator? generator=None) -> (Tensor)
        aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> (Tensor)
        aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> (Tensor)
        aten::normal.functional(Tensor self, float mean=0., float std=1., *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenNotEqualSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::not_equal.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::not_equal.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenNuclearNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::nuclear_norm(Tensor self, bool keepdim=False) -> (Tensor)
        aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> (Tensor)'''
        pass


class ATenNumpyTSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::numpy_T(Tensor(a) self) -> (Tensor(a))
        aten::numpy_T.a(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenOneHotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::one_hot(Tensor self, int num_classes=-1) -> (Tensor)'''
        pass


class ATenOnesLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ones_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenOrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__or__.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::__or__.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenOrgqrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::orgqr(Tensor self, Tensor input2) -> (Tensor)'''
        pass


class ATenOrmqrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> (Tensor)'''
        pass


class ATenOuterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::outer(Tensor self, Tensor vec2) -> (Tensor)'''
        pass


class ATenPackPaddedSequenceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)'''
        pass


class ATenPackSequenceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_pack_sequence(Tensor output, Tensor batch_sizes, Tensor? sorted_indices, Tensor? unsorted_indices) -> (Tensor, Tensor, Tensor?, Tensor?)'''
        pass


class ATenPadSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pad(Tensor self, int[] pad, str mode="constant", float? value=None) -> (Tensor)'''
        pass


class ATenPadCircularSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_pad_circular(Tensor self, int[] pad) -> (Tensor)'''
        pass


class ATenPadEnumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_pad_enum(Tensor self, int[] pad, int mode, float? value=None) -> (Tensor)'''
        pass


class ATenPadPackedSequenceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)'''
        pass


class ATenPadSequenceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.) -> (Tensor)'''
        pass


class ATenPairwiseDistanceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pairwise_distance(Tensor x1, Tensor x2, float p=2., float eps=9.9999999999999995e-07, bool keepdim=False) -> (Tensor)'''
        pass


class ATenPdistSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pdist(Tensor self, float p=2.) -> (Tensor)'''
        pass


class ATenPermuteSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::permute(Tensor(a) self, int[] dims) -> (Tensor(a))'''
        pass


class ATenPermuteCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::permute_copy(Tensor self, int[] dims) -> (Tensor)'''
        pass


class ATenPinMemorySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pin_memory(Tensor(a) self, Device? device=None) -> (Tensor(a))
        aten::_pin_memory(Tensor self, Device? device=None) -> (Tensor)'''
        pass


class ATenPinverseSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pinverse(Tensor self, float rcond=1.0000000000000001e-15) -> (Tensor)'''
        pass


class ATenPixelShuffleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pixel_shuffle(Tensor self, int upscale_factor) -> (Tensor)'''
        pass


class ATenPixelUnshuffleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pixel_unshuffle(Tensor self, int downscale_factor) -> (Tensor)'''
        pass


class ATenPoissonSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::poisson(Tensor self, Generator? generator=None) -> (Tensor)'''
        pass


class ATenPoissonNllLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> (Tensor)'''
        pass


class ATenPolarSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::polar(Tensor abs, Tensor angle) -> (Tensor)'''
        pass


class ATenPolygammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::polygamma(int n, Tensor self) -> (Tensor)'''
        pass


class ATenPositiveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::positive(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenPowSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)
        aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> (Tensor)
        aten::pow.Scalar(Scalar self, Tensor exponent) -> (Tensor)'''
        pass


class ATenPreluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::prelu(Tensor self, Tensor weight) -> (Tensor)'''
        pass


class ATenProdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::prod(Tensor self, *, int? dtype=None) -> (Tensor)
        aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)
        aten::prod.dim_Dimname(Tensor self, str dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenPutSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> (Tensor)'''
        pass


class ATenQPerChannelScalesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::q_per_channel_scales(Tensor self) -> (Tensor)'''
        pass


class ATenQPerChannelZeroPointsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::q_per_channel_zero_points(Tensor self) -> (Tensor)'''
        pass


class ATenQrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
        aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)'''
        pass


class ATenQuantileSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> (Tensor)
        aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> (Tensor)'''
        pass


class ATenQuantizePerChannelSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, int dtype) -> (Tensor)'''
        pass


class ATenQuantizePerTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantize_per_tensor(Tensor self, float scale, int zero_point, int dtype) -> (Tensor)
        aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int dtype) -> (Tensor)
        aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, int dtype) -> (Tensor[])'''
        pass


class ATenQuantizePerTensorDynamicSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantize_per_tensor_dynamic(Tensor self, int dtype, bool reduce_range) -> (Tensor)'''
        pass


class ATenQuantizedBatchNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> (Tensor)'''
        pass


class ATenQuantizedGruSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_gru.input(Tensor input, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
        aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
        aten::quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
        aten::quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)'''
        pass


class ATenQuantizedGruCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor)'''
        pass


class ATenQuantizedLstmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_lstm.input(Tensor input, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
        aten::quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
        aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
        aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, int? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenQuantizedLstmCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)'''
        pass


class ATenQuantizedMaxPool1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> (Tensor)'''
        pass


class ATenQuantizedMaxPool2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False) -> (Tensor)'''
        pass


class ATenQuantizedRnnReluCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor)'''
        pass


class ATenQuantizedRnnTanhCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor)'''
        pass


class ATenRad2degSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rad2deg(Tensor self) -> (Tensor)'''
        pass


class ATenRandLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenRandintLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)
        aten::randint_like.low_dtype(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenRandnLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass


class ATenRandomSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::random.from_functional(Tensor self, int from, int? to, *, Generator? generator=None) -> (Tensor)
        aten::random.to_functional(Tensor self, int to, *, Generator? generator=None) -> (Tensor)
        aten::random.functional(Tensor self, *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenRavelSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::ravel(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenRealSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::real(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenReciprocalSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::reciprocal(Tensor self) -> (Tensor)'''
        pass


class ATenRefineNamesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::refine_names(Tensor(a) self, str[] names) -> (Tensor(a))'''
        pass


class ATenReflectionPad1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::reflection_pad1d(Tensor self, int[2] padding) -> (Tensor)'''
        pass


class ATenReflectionPad2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::reflection_pad2d(Tensor self, int[4] padding) -> (Tensor)'''
        pass


class ATenReflectionPad3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::reflection_pad3d(Tensor self, int[6] padding) -> (Tensor)'''
        pass


class ATenReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::relu(Tensor self) -> (Tensor)'''
        pass


class ATenRelu6Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::relu6(Tensor self) -> (Tensor)'''
        pass


class ATenRemainderSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::remainder.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> (Tensor)
        aten::remainder.Scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenRemoveBatchDimSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_remove_batch_dim(Tensor self, int level, int batch_size, int out_dim) -> (Tensor)'''
        pass


class ATenRenameSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rename(Tensor(a) self, str[]? names) -> (Tensor(a))'''
        pass


class ATenRenormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> (Tensor)'''
        pass


class ATenRepeatSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::repeat(Tensor self, int[] repeats) -> (Tensor)'''
        pass


class ATenRepeatInterleaveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> (Tensor)
        aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> (Tensor)
        aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> (Tensor)'''
        pass


class ATenReplicationPad1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::replication_pad1d(Tensor self, int[2] padding) -> (Tensor)'''
        pass


class ATenReplicationPad2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::replication_pad2d(Tensor self, int[4] padding) -> (Tensor)'''
        pass


class ATenReplicationPad3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::replication_pad3d(Tensor self, int[6] padding) -> (Tensor)'''
        pass


class ATenReshapeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::reshape(Tensor(a) self, int[] shape) -> (Tensor(a))'''
        pass


class ATenReshapeAliasSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_reshape_alias(Tensor(a) self, int[] size, int[] stride) -> (Tensor(a))'''
        pass


class ATenReshapeAliasCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_reshape_alias_copy(Tensor self, int[] size, int[] stride) -> (Tensor)'''
        pass


class ATenReshapeAsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::reshape_as(Tensor(a) self, Tensor other) -> (Tensor(a))'''
        pass


class ATenReshapeFromTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_reshape_from_tensor(Tensor self, Tensor shape) -> (Tensor)'''
        pass


class ATenResizeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::resize.functional(Tensor self, int[] size, *, int? memory_format=None) -> (Tensor)'''
        pass


class ATenResizeAsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::resize_as.functional(Tensor self, Tensor the_template, *, int? memory_format=None) -> (Tensor)'''
        pass


class ATenResizeOutputSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_resize_output.functional(Tensor self, int[] size, Device device) -> (Tensor)'''
        pass


class ATenResolveConjSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::resolve_conj(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenResolveNegSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::resolve_neg(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenRnnReluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
        aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)'''
        pass


class ATenRnnReluCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor)'''
        pass


class ATenRnnTanhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
        aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)'''
        pass


class ATenRnnTanhCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor)'''
        pass


class ATenRollSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)'''
        pass


class ATenRot90Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rot90(Tensor self, int k=1, int[] dims=[0, 1]) -> (Tensor)'''
        pass


class ATenRoundSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::round(Tensor self) -> (Tensor)
        aten::round.decimals(Tensor self, *, int decimals) -> (Tensor)'''
        pass


class ATenRowIndicesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::row_indices(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenRowIndicesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::row_indices_copy(Tensor self) -> (Tensor)'''
        pass


class ATenRowStackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::row_stack(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenRowwisePruneSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_rowwise_prune(Tensor weight, Tensor mask, int compressed_indices_dtype) -> (Tensor, Tensor)'''
        pass


class ATenRreluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> (Tensor)'''
        pass


class ATenRreluWithNoiseSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> (Tensor)'''
        pass


class ATenRshiftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__rshift__.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::__rshift__.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenRsqrtSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rsqrt(Tensor self) -> (Tensor)'''
        pass


class ATenRsubSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
        aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenSampleDirichletSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> (Tensor)'''
        pass


class ATenScatterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> (Tensor)
        aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
        aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> (Tensor)
        aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> (Tensor)
        aten::scatter.dimname_src(Tensor self, str dim, Tensor index, Tensor src) -> (Tensor)
        aten::scatter.dimname_value(Tensor self, str dim, Tensor index, Scalar value) -> (Tensor)'''
        pass


class ATenScatterAddSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> (Tensor)
        aten::scatter_add.dimname(Tensor self, str dim, Tensor index, Tensor src) -> (Tensor)'''
        pass


class ATenScatterReduceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> (Tensor)'''
        pass


class ATenSearchsortedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> (Tensor)
        aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> (Tensor)'''
        pass


class ATenSegmentReduceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> (Tensor)'''
        pass


class ATenSelectSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::select.int(Tensor(a) self, int dim, int index) -> (Tensor(a))
        aten::select.Dimname(Tensor(a) self, str dim, int index) -> (Tensor(a))'''
        pass


class ATenSelectCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::select_copy.int(Tensor self, int dim, int index) -> (Tensor)'''
        pass


class ATenSelectScatterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::select_scatter(Tensor self, Tensor src, int dim, int index) -> (Tensor)'''
        pass


class ATenSeluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::selu(Tensor self) -> (Tensor)'''
        pass


class ATenSetSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::set.source_Storage_functional(Tensor self, Storage source) -> (Tensor)
        aten::set.source_Storage_storage_offset_functional(Tensor self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> (Tensor)
        aten::set.source_Tensor_functional(Tensor self, Tensor source) -> (Tensor)
        aten::set.functional(Tensor self) -> (Tensor)'''
        pass


class ATenSgnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sgn(Tensor self) -> (Tensor)'''
        pass


class ATenShapeAsTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_shape_as_tensor(Tensor self) -> (Tensor)'''
        pass


class ATenSigmoidSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sigmoid(Tensor self) -> (Tensor)'''
        pass


class ATenSignSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sign(Tensor self) -> (Tensor)'''
        pass


class ATenSignbitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::signbit(Tensor self) -> (Tensor)'''
        pass


class ATenSiluSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::silu(Tensor self) -> (Tensor)'''
        pass


class ATenSinSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sin(Tensor self) -> (Tensor)'''
        pass


class ATenSincSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sinc(Tensor self) -> (Tensor)'''
        pass


class ATenSinhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sinh(Tensor self) -> (Tensor)'''
        pass


class ATenSizeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::size(Tensor self) -> (int[])'''
        pass


class ATenSliceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor(a))'''
        pass


class ATenSliceCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slice_copy.Tensor(Tensor self, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor)'''
        pass


class ATenSliceScatterSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slice_scatter(Tensor self, Tensor src, int dim=0, int? start=None, int? end=None, int step=1) -> (Tensor)'''
        pass


class ATenSlogdetSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)'''
        pass


class ATenSlowConv3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=[1, 1, 1], int[3] padding=[0, 0, 0]) -> (Tensor)'''
        pass


class ATenSlowConvDilated2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=[1, 1], int[2] padding=[0, 0], int[2] dilation=[1, 1]) -> (Tensor)'''
        pass


class ATenSlowConvDilated3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=[1, 1, 1], int[3] padding=[0, 0, 0], int[3] dilation=[1, 1, 1]) -> (Tensor)'''
        pass


class ATenSlowConvTranspose2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=[1, 1], int[2] padding=[0, 0], int[2] output_padding=[0, 0], int[2] dilation=[1, 1]) -> (Tensor)'''
        pass


class ATenSlowConvTranspose3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=[1, 1, 1], int[3] padding=[0, 0, 0], int[3] output_padding=[0, 0, 0], int[3] dilation=[1, 1, 1]) -> (Tensor)'''
        pass


class ATenSmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::smm(Tensor self, Tensor mat2) -> (Tensor)'''
        pass


class ATenSmoothL1LossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=1, float beta=1.) -> (Tensor)'''
        pass


class ATenSobolEngineDrawSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, int? dtype) -> (Tensor, Tensor)'''
        pass


class ATenSoftMarginLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::soft_margin_loss(Tensor self, Tensor target, int reduction=1) -> (Tensor)'''
        pass


class ATenSoftmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::softmax.int(Tensor self, int dim, int? dtype=None) -> (Tensor)
        aten::softmax.Dimname(Tensor self, str dim, *, int? dtype=None) -> (Tensor)
        aten::_softmax(Tensor self, int dim, bool half_to_float) -> (Tensor)'''
        pass


class ATenSoftmaxBackwardDataSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, int input_dtype) -> (Tensor)'''
        pass


class ATenSoftplusSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> (Tensor)'''
        pass


class ATenSoftshrinkSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::softshrink(Tensor self, Scalar lambd=0.5) -> (Tensor)'''
        pass


class ATenSortSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
        aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
        aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::sort.dimname(Tensor self, str dim, bool descending=False) -> (Tensor values, Tensor indices)
        aten::sort.dimname_values(Tensor self, str dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        aten::sort.dimname_stable(Tensor self, *, bool? stable, str dim, bool descending=False) -> (Tensor values, Tensor indices)
        aten::sort.dimname_values_stable(Tensor self, *, bool? stable, str dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenSortedSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sorted.Tensor(Tensor[](a) input) -> (Tensor[])'''
        pass


class ATenSpecialDigammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_digamma(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialEntrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_entr(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialErfSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_erf(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialErfcSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_erfc(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialErfcxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_erfcx(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialErfinvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_erfinv(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialExp2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_exp2(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialExpitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_expit(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialExpm1Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_expm1(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialGammaincSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_gammainc(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenSpecialGammainccSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_gammaincc(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenSpecialGammalnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_gammaln(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialI0Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_i0(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialI0eSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_i0e(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialI1Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_i1(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialI1eSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_i1e(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialLog1pSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_log1p(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialLogNdtrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_log_ndtr(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialLogSoftmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_log_softmax(Tensor self, int dim, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenSpecialLogitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_logit(Tensor self, float? eps=None) -> (Tensor)'''
        pass


class ATenSpecialLogsumexpSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> (Tensor)'''
        pass


class ATenSpecialMultigammalnSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_multigammaln(Tensor self, int p) -> (Tensor)'''
        pass


class ATenSpecialNdtrSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_ndtr(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialNdtriSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_ndtri(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialPolygammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_polygamma(int n, Tensor self) -> (Tensor)'''
        pass


class ATenSpecialPsiSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_psi(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialRoundSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_round(Tensor self, *, int decimals=0) -> (Tensor)'''
        pass


class ATenSpecialSincSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_sinc(Tensor self) -> (Tensor)'''
        pass


class ATenSpecialSoftmaxSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_softmax(Tensor self, int dim, int? dtype=None) -> (Tensor)'''
        pass


class ATenSpecialXlog1pySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_xlog1py(Tensor self, Tensor other) -> (Tensor)
        aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> (Tensor)
        aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenSpecialXlogySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_xlogy(Tensor self, Tensor other) -> (Tensor)
        aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> (Tensor)
        aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenSpecialZetaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::special_zeta(Tensor self, Tensor other) -> (Tensor)
        aten::special_zeta.self_scalar(Scalar self, Tensor other) -> (Tensor)
        aten::special_zeta.other_scalar(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenSplitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> (Tensor[])
        aten::split.sizes(Tensor(a -> *) self, int[] split_size, int dim=0) -> (Tensor[])
        aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> (Tensor[])'''
        pass


class ATenSplitCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::split_copy.Tensor(Tensor self, int split_size, int dim=0) -> (Tensor[])'''
        pass


class ATenSplitWithSizesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> (Tensor[])'''
        pass


class ATenSplitWithSizesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::split_with_sizes_copy(Tensor self, int[] split_sizes, int dim=0) -> (Tensor[])'''
        pass


class ATenSqrtSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sqrt(Tensor self) -> (Tensor)'''
        pass


class ATenSquareSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::square(Tensor self) -> (Tensor)'''
        pass


class ATenSqueezeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::squeeze(Tensor(a) self) -> (Tensor(a))
        aten::squeeze.dim(Tensor(a) self, int dim) -> (Tensor(a))
        aten::squeeze.dimname(Tensor(a) self, str dim) -> (Tensor(a))'''
        pass


class ATenSqueezeCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::squeeze_copy(Tensor self) -> (Tensor)
        aten::squeeze_copy.dim(Tensor self, int dim) -> (Tensor)'''
        pass


class ATenSspaddmmSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenStackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::stack(Tensor[] tensors, int dim=0) -> (Tensor)
        aten::_stack(Tensor[] tensors, int dim=0) -> (Tensor)'''
        pass


class ATenStandardGammaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_standard_gamma(Tensor self, Generator? generator=None) -> (Tensor)'''
        pass


class ATenStdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::std(Tensor self, bool unbiased=True) -> (Tensor)
        aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor)
        aten::std.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor)
        aten::std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor)
        aten::std.correction_names(Tensor self, str[1] dim, *, int? correction, bool keepdim=False) -> (Tensor)'''
        pass


class ATenStdMeanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
        aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
        aten::std_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
        aten::std_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
        aten::std_mean.correction_names(Tensor self, str[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)'''
        pass


class ATenStftSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> (Tensor)
        aten::stft.center(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, str pad_mode="reflect", bool normalized=False, bool? onesided=None, bool? return_complex=None) -> (Tensor)'''
        pass


class ATenSubSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
        aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenSubtractSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
        aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)'''
        pass


class ATenSumSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)
        aten::sum(Tensor self, *, int? dtype=None) -> (Tensor)
        aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)'''
        pass


class ATenSumToSizeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::sum_to_size(Tensor self, int[] size) -> (Tensor)'''
        pass


class ATenSvdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
        aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)'''
        pass


class ATenSwapaxesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> (Tensor(a))'''
        pass


class ATenSwapdimsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::swapdims(Tensor(a) self, int dim0, int dim1) -> (Tensor(a))'''
        pass


class ATenSymeigSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
        aten::symeig.e(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)'''
        pass


class ATenSymeigHelperSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)'''
        pass


class ATenTSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::t(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenTCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::t_copy(Tensor self) -> (Tensor)'''
        pass


class ATenTakeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::take(Tensor self, Tensor index) -> (Tensor)'''
        pass


class ATenTakeAlongDimSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> (Tensor)'''
        pass


class ATenTanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::tan(Tensor self) -> (Tensor)'''
        pass


class ATenTanhSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::tanh(Tensor self) -> (Tensor)'''
        pass


class ATenTensorSplitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::tensor_split.sections(Tensor(a -> *) self, int sections, int dim=0) -> (Tensor[])
        aten::tensor_split.indices(Tensor(a -> *) self, int[] indices, int dim=0) -> (Tensor[])
        aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> (Tensor[])'''
        pass


class ATenTensorToListSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_tensor_to_list(Tensor self) -> (int[])'''
        pass


class ATenTensordotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> (Tensor)'''
        pass


class ATenThnnConv2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=[1, 1], int[2] padding=[0, 0]) -> (Tensor)'''
        pass


class ATenThnnFusedGruCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)'''
        pass


class ATenThnnFusedLstmCellSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenThnnFusedLstmCellBackwardImplSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_thnn_fused_lstm_cell_backward_impl(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenThresholdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::threshold(Tensor self, Scalar threshold, Scalar value) -> (Tensor)'''
        pass


class ATenTileSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::tile(Tensor self, int[] dims) -> (Tensor)'''
        pass


class ATenToSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))
        aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))
        aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))
        aten::to.dtype_layout(Tensor(a) self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))
        aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))
        aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))
        aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))'''
        pass


class ATenToCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_to_copy(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, int? memory_format=None) -> (Tensor)'''
        pass


class ATenToCpuSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_to_cpu(Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenToDenseSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_to_dense(Tensor self, int? dtype=None) -> (Tensor)
        aten::to_dense(Tensor self, int? dtype=None) -> (Tensor)'''
        pass


class ATenToPaddedTensorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::to_padded_tensor(Tensor self, float padding, int[]? output_size=None) -> (Tensor)'''
        pass


class ATenTopkSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
        aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)'''
        pass


class ATenTraceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::trace(Tensor self) -> (Tensor)'''
        pass


class ATenTransformBiasRescaleQkvSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias, int num_heads) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenTransformerEncoderLayerFwdSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_transformer_encoder_layer_fwd(Tensor src, int embed_dim, int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, bool use_gelu, bool norm_first, float eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None) -> (Tensor)'''
        pass


class ATenTransposeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> (Tensor(a))
        aten::transpose.Dimname(Tensor(a) self, str dim0, str dim1) -> (Tensor(a))'''
        pass


class ATenTransposeCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::transpose_copy.int(Tensor self, int dim0, int dim1) -> (Tensor)'''
        pass


class ATenTrapezoidSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> (Tensor)
        aten::trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> (Tensor)'''
        pass


class ATenTrapzSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> (Tensor)
        aten::trapz.dx(Tensor y, *, float dx=1., int dim=-1) -> (Tensor)'''
        pass


class ATenTriangularSolveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
        aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)'''
        pass


class ATenTrilSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::tril(Tensor self, int diagonal=0) -> (Tensor)'''
        pass


class ATenTrilinearSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> (Tensor)'''
        pass


class ATenTripletMarginLossSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1., float p=2., float eps=9.9999999999999995e-07, bool swap=False, int reduction=1) -> (Tensor)'''
        pass


class ATenTriuSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::triu(Tensor self, int diagonal=0) -> (Tensor)'''
        pass


class ATenTrueDivideSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::true_divide.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::true_divide.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenTruncSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::trunc(Tensor self) -> (Tensor)'''
        pass


class ATenTypeAsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::type_as(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenUnbindSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unbind.int(Tensor(a -> *) self, int dim=0) -> (Tensor[])
        aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> (Tensor[])'''
        pass


class ATenUnbindCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unbind_copy.int(Tensor self, int dim=0) -> (Tensor[])'''
        pass


class ATenUnflattenSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unflatten.int(Tensor(a) self, int dim, int[] sizes, str[]? names=None) -> (Tensor(a))
        aten::unflatten.Dimname(Tensor(a) self, str dim, int[] sizes, str[] names) -> (Tensor(a))'''
        pass


class ATenUnflattenDenseTensorsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> (Tensor[])'''
        pass


class ATenUnfoldSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unfold(Tensor(a) self, int dimension, int size, int step) -> (Tensor(a))'''
        pass


class ATenUnfoldCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unfold_copy(Tensor self, int dimension, int size, int step) -> (Tensor)'''
        pass


class ATenUniformSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::uniform.functional(Tensor self, float from=0., float to=1., *, Generator? generator=None) -> (Tensor)'''
        pass


class ATenUniqueSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)'''
        pass


class ATenUnique2Schema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenUniqueConsecutiveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenUniqueDimSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenUniqueDimConsecutiveSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)'''
        pass


class ATenUnpackDualSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)'''
        pass


class ATenUnsafeChunkSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> (Tensor[])'''
        pass


class ATenUnsafeSplitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unsafe_split.Tensor(Tensor self, int split_size, int dim=0) -> (Tensor[])'''
        pass


class ATenUnsafeSplitWithSizesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unsafe_split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> (Tensor[])'''
        pass


class ATenUnsafeViewSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_unsafe_view(Tensor self, int[] size) -> (Tensor)'''
        pass


class ATenUnsqueezeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unsqueeze(Tensor(a) self, int dim) -> (Tensor(a))'''
        pass


class ATenUnsqueezeCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::unsqueeze_copy(Tensor self, int dim) -> (Tensor)'''
        pass


class ATenUpsampleSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__upsample(Tensor input, int? size=None, int? scale_factor=None, str mode="nearest", bool? align_corners=None) -> (Tensor)
        aten::__upsample.size_list(Tensor input, int[]? size=None, int? scale_factor=None, str mode="nearest", bool? align_corners=None) -> (Tensor)'''
        pass


class ATenUpsampleBicubic2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::upsample_bicubic2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleBicubic2dAaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_upsample_bicubic2d_aa(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::_upsample_bicubic2d_aa.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleBilinearSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__upsample_bilinear(Tensor input, int? size=None, int? scale_factor=None) -> (Tensor)
        aten::__upsample_bilinear.size_list(Tensor input, int[]? size=None, int? scale_factor=None) -> (Tensor)
        aten::__upsample_bilinear.scale_list(Tensor input, int? size=None, int[]? scale_factor=None) -> (Tensor)
        aten::__upsample_bilinear.size_list_scale_list(Tensor input, int[]? size=None, int[]? scale_factor=None) -> (Tensor)'''
        pass


class ATenUpsampleBilinear2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleBilinear2dAaSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_upsample_bilinear2d_aa(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::_upsample_bilinear2d_aa.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleLinear1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> (Tensor)
        aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleNearestSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__upsample_nearest(Tensor input, int? size=None, int? scale_factor=None) -> (Tensor)
        aten::__upsample_nearest.size_list(Tensor input, int[]? size=None, int? scale_factor=None) -> (Tensor)'''
        pass


class ATenUpsampleNearest1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> (Tensor)
        aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleNearest2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleNearest3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)
        aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> (Tensor)'''
        pass


class ATenUpsampleNearestExact1dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_upsample_nearest_exact1d(Tensor self, int[1] output_size, float? scales=None) -> (Tensor)
        aten::_upsample_nearest_exact1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleNearestExact2dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_upsample_nearest_exact2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::_upsample_nearest_exact2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenUpsampleNearestExact3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_upsample_nearest_exact3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)
        aten::_upsample_nearest_exact3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> (Tensor)'''
        pass


class ATenUpsampleTrilinear3dSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> (Tensor)
        aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> (Tensor)'''
        pass


class ATenValuesSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_values(Tensor(a) self) -> (Tensor(a))
        aten::values(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenValuesCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::values_copy(Tensor self) -> (Tensor)
        aten::_values_copy(Tensor self) -> (Tensor)'''
        pass


class ATenVanderSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::vander(Tensor x, int? N=None, bool increasing=False) -> (Tensor)'''
        pass


class ATenVarSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::var(Tensor self, bool unbiased=True) -> (Tensor)
        aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor)
        aten::var.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor)
        aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor)
        aten::var.correction_names(Tensor self, str[1] dim, *, int? correction, bool keepdim=False) -> (Tensor)'''
        pass


class ATenVarMeanSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
        aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
        aten::var_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
        aten::var_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
        aten::var_mean.correction_names(Tensor self, str[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)'''
        pass


class ATenVdotSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::vdot(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenViewSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view(Tensor(a) self, int[] size) -> (Tensor(a))
        aten::view.dtype(Tensor(a) self, int dtype) -> (Tensor(a))'''
        pass


class ATenViewAsSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view_as(Tensor(a) self, Tensor other) -> (Tensor(a))'''
        pass


class ATenViewAsComplexSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view_as_complex(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenViewAsComplexCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view_as_complex_copy(Tensor self) -> (Tensor)'''
        pass


class ATenViewAsRealSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view_as_real(Tensor(a) self) -> (Tensor(a))'''
        pass


class ATenViewAsRealCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view_as_real_copy(Tensor self) -> (Tensor)'''
        pass


class ATenViewCopySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::view_copy(Tensor self, int[] size) -> (Tensor)
        aten::view_copy.dtype(Tensor self, int dtype) -> (Tensor)'''
        pass


class ATenVsplitSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::vsplit.int(Tensor(a -> *) self, int sections) -> (Tensor[])
        aten::vsplit.array(Tensor(a -> *) self, int[] indices) -> (Tensor[])'''
        pass


class ATenVstackSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::vstack(Tensor[] tensors) -> (Tensor)'''
        pass


class ATenWeightNormSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> (Tensor)'''
        pass


class ATenWeightNormInterfaceSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::_weight_norm_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)'''
        pass


class ATenWhereSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::where.self(Tensor condition, Tensor self, Tensor other) -> (Tensor)
        aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> (Tensor)
        aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> (Tensor)
        aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> (Tensor)
        aten::where(Tensor condition) -> (Tensor[])'''
        pass


class ATenXlogySchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::xlogy.Tensor(Tensor self, Tensor other) -> (Tensor)
        aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> (Tensor)
        aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> (Tensor)'''
        pass


class ATenXorSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::__xor__.Scalar(Tensor self, Scalar other) -> (Tensor)
        aten::__xor__.Tensor(Tensor self, Tensor other) -> (Tensor)'''
        pass


class ATenZeroSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::zero.functional(Tensor self) -> (Tensor)'''
        pass


class ATenZerosLikeSchema(OperatorConverter):
    @abstractmethod
    def parse(self, node, attrs, args, graph_converter):
        '''aten::zeros_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, int? memory_format=None) -> (Tensor)'''
        pass
