#![allow(dead_code)]

// RTMPose-m estimator (SimCC head, 256x192 input). Tensor names match the
// canonical mmpose ONNX export via mmdeploy's
// `pose-detection_simcc_onnxruntime_dynamic.py` config. Verify against the
// shipped ONNX with netron after upload and adjust if they differ.
pub const RTMPOSE_M_MODEL_NAME: &str = "rtmpose_m_simcc_aic_coco_256x192.onnx";
pub const RTMPOSE_L_MODEL_NAME: &str = "rtmpose_l_simcc_aic_coco_256x192.onnx";

pub const RTMPOSE_INPUT_TENSOR_NAME: &str = "input";
pub const RTMPOSE_SIMCC_X_OUTPUT: &str = "simcc_x";
pub const RTMPOSE_SIMCC_Y_OUTPUT: &str = "simcc_y";

// RTMPose-m / -l canonical config: 256 high, 192 wide, NCHW.
pub const RTMPOSE_INPUT_WIDTH: u32 = 192;
pub const RTMPOSE_INPUT_HEIGHT: u32 = 256;
pub const RTMPOSE_INPUT_DIMENSIONS: (u32, u32) = (RTMPOSE_INPUT_WIDTH, RTMPOSE_INPUT_HEIGHT);
pub const RTMPOSE_INPUT_TENSOR_SHAPE: (usize, usize, usize, usize) = (
    1,
    3,
    RTMPOSE_INPUT_HEIGHT as usize,
    RTMPOSE_INPUT_WIDTH as usize,
);

// SimCC representation: per-axis 1-D classification head, with `bins = dim *
// split_ratio`. 2.0 is the upstream default that gives sub-pixel accuracy
// without bloating the output.
pub const RTMPOSE_SIMCC_SPLIT_RATIO: f32 = 2.0;

// ImageNet normalization, scaled to the 0..255 input range so it composes with
// the shared `raw_rgb_to_normalized_tensor::MeanStd` path (which expects mean
// and std on the same scale as the raw pixels).
pub const IMAGENET_MEAN_0_255: [f32; 3] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0];
pub const IMAGENET_STD_0_255: [f32; 3] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0];

// Canonical COCO-17 topology now lives in `shared::topology`; re-export so the
// RTM estimator/visualizer keep their existing `config::` references.
pub use crate::shared::topology::{COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES, N_COCO_KEYPOINTS};

// Per-keypoint confidence threshold below which a landmark is treated as not
// detected (drawing and ROI derivation skip it).
pub const RTMPOSE_LANDMARK_CONFIDENCE_THRESHOLD: f32 = 0.3;

// RTMDet-nano person detector. 320x320 input, NCHW, ImageNet-style mean/std
// normalization in the upstream deploy config. Tensor names mirror the
// canonical `detection_onnxruntime_static.py` mmdeploy export — `dets`
// concatenates [x1, y1, x2, y2, score] and `labels` carries the COCO class id.
pub const RTMDET_NANO_MODEL_NAME: &str = "rtmdet_nano_person.onnx";
pub const RTMDET_INPUT_TENSOR_NAME: &str = "input";
pub const RTMDET_OUTPUT_DETS: &str = "dets";
pub const RTMDET_OUTPUT_LABELS: &str = "labels";
pub const RTMDET_INPUT_SIZE: u32 = 320;
pub const RTMDET_INPUT_DIMENSIONS: (u32, u32) = (RTMDET_INPUT_SIZE, RTMDET_INPUT_SIZE);
pub const RTMDET_INPUT_TENSOR_SHAPE: (usize, usize, usize, usize) =
    (1, 3, RTMDET_INPUT_SIZE as usize, RTMDET_INPUT_SIZE as usize);

// mmdetection ships rtmdet pretrained on COCO; class 0 is `person`.
pub const COCO_PERSON_CLASS_ID: i64 = 0;

// Score threshold for accepting a detection. Matches the upstream default in
// mmdeploy's `_base_/static.py`.
pub const RTMDET_SCORE_THRESHOLD: f32 = 0.45;
pub const RTMDET_NMS_IOU_THRESHOLD: f32 = 0.65;
