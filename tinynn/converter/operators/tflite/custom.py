from .generated_ops import CustomOperator

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
    def __init__(self, inputs, outputs, max_detections: int, max_classes_per_detection: int,
                 nms_score_threshold: float, nms_iou_threshold: float,
                 num_classes: int, y_scale: float, x_scale: float,
                 h_scale: float, w_scale: float) -> None:
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
            fbb.Int('num_classes', self.nms_iou_threshold)
            fbb.Float('y_scale', self.y_scale)
            fbb.Float('y_scale', self.x_scale)
            fbb.Float('y_scale', self.h_scale)
            fbb.Float('y_scale', self.w_scale)
        self.custom_options = fbb.Finish()
        return super().build(builder)
