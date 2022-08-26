from .generated_ops import CustomOperator
from .base import create_byte_array
from ...schemas.tflite import schema_generated as tflite

HAS_FLEXBUFFER = False
try:
    from flatbuffers import flexbuffers

    HAS_FLEXBUFFER = True
except ImportError:
    pass


class Atan2Operator(CustomOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(inputs, outputs)
        self.op.custom_code = "Atan2"


class TFLiteDetectionPostprocessOperator(CustomOperator):
    def __init__(
        self,
        inputs,
        outputs,
        max_detections: int,
        max_classes_per_detection: int,
        nms_score_threshold: float,
        nms_iou_threshold: float,
        num_classes: int,
        y_scale: float,
        x_scale: float,
        h_scale: float,
        w_scale: float,
    ) -> None:
        super().__init__(inputs, outputs)
        assert HAS_FLEXBUFFER, "TFLITE_DETECTION_POSTPROCESS relies on FlexBuffer, which requires flatbuffers>=2"
        self.op.custom_code = "TFLITE_DETECTION_POSTPROCESS"
        self.max_detections = max_detections
        self.max_classes_per_detection = max_classes_per_detection
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.num_classes = num_classes
        self.y_scale = y_scale
        self.x_scale = x_scale
        self.h_scale = h_scale
        self.w_scale = w_scale

    def build(self, builder):
        fbb = flexbuffers.Builder()
        with fbb.Map():
            fbb.Int('max_detections', self.max_detections)
            fbb.Int('max_classes_per_detection', self.max_classes_per_detection)
            fbb.Float('nms_score_threshold', self.nms_score_threshold)
            fbb.Float('nms_iou_threshold', self.nms_iou_threshold)
            fbb.Int('num_classes', self.num_classes)
            fbb.Float('y_scale', self.y_scale)
            fbb.Float('x_scale', self.x_scale)
            fbb.Float('h_scale', self.h_scale)
            fbb.Float('w_scale', self.w_scale)
        self.custom_options = fbb.Finish()
        return super().build(builder)


class MTKTransposeConvOperator(CustomOperator):
    def __init__(
        self,
        inputs,
        outputs,
        activation: int = tflite.ActivationFunctionType.NONE,
        depth_multiplier: int = 0,
        dilation_height_factor: int = 0,
        dilation_width_factor: int = 0,
        padding_type: int = tflite.Padding.SAME,
        stride_height: int = 0,
        stride_width: int = 0,
    ) -> None:
        super().__init__(inputs, outputs)
        self.op.custom_code = "MTK_TRANSPOSE_CONV"
        self.activation = activation
        self.depth_multiplier = depth_multiplier
        self.dilation_height_factor = dilation_height_factor
        self.dilation_width_factor = dilation_width_factor
        self.padding_type = padding_type
        self.stride_height = stride_height
        self.stride_width = stride_width

    def build(self, builder):
        fbb = flexbuffers.Builder()
        fbb.MapFromElements(
            {
                'activation': self.activation,
                'depth_multiplier': self.depth_multiplier,
                'dilation_height_factor': self.dilation_height_factor,
                'dilation_width_factor': self.dilation_width_factor,
                'PaddingType': self.padding_type,
                'stride_height': self.stride_height,
                'stride_width': self.stride_width,
            }
        )
        self.custom_options = fbb.Finish()
        return super().build(builder)
