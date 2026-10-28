use anyhow::Result;
use image::DynamicImage;
use std::fmt::Display;
use std::path::PathBuf;

// From processing/types.rs
#[derive(Debug, Clone, Copy)]
pub enum BorderMode {
    Constant(u8),
    Replicate,
}

pub enum NormalizationType {
    UnitScale,
    // 0..1 normalization that respects NHWC layout (palm detector)
    UnitRange,
    // linearly maps pixels from 0:255 to -1:1
    Linear,
    MeanStd(MeanStdNormalizationConfig),
}

pub struct MeanStdNormalizationConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

pub struct DetectorOutput {
    pub regions_of_interest: Vec<RegionOfInterest>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RegionOfInterest {
    pub bbox: Bbox,
    pub confidence: f32,
    pub landmarks: Option<Vec<Landmark2D>>,
    pub rotation: Option<f32>, // in radians
}

impl RegionOfInterest {
    pub fn reverse_padding_and_scaling(&mut self, padding: (u32, u32), scale: f32) {
        self.bbox.reverse_padding_and_scaling(padding, scale);
        if let Some(landmarks) = &mut self.landmarks {
            for landmark in landmarks {
                landmark.reverse_padding_and_scaling(padding, scale);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Bbox {
    BboxXYXY(BboxXYXY),
    BboxXYWH(BboxXYWH),
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct BboxXYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct BboxXYWH {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
}

impl BboxXYXY {
    fn reverse_padding_and_scaling(&mut self, padding: (u32, u32), scale: f32) {
        // First, remove padding offset
        self.x1 -= padding.0 as f32;
        self.y1 -= padding.1 as f32;
        self.x2 -= padding.0 as f32;
        self.y2 -= padding.1 as f32;

        // Then, scale back to original size
        self.x1 /= scale;
        self.y1 /= scale;
        self.x2 /= scale;
        self.y2 /= scale;
    }
}

impl BboxXYXY {
    fn scale_uniformly(&mut self, factor: f32, max_xy: Option<(f32, f32)>) {
        // Calculate current center and dimensions
        let center_x = (self.x1 + self.x2) / 2.0;
        let center_y = (self.y1 + self.y2) / 2.0;
        let width = (self.x2 - self.x1).abs() * factor;
        let height = (self.y2 - self.y1).abs() * factor;

        // Calculate new corners after scaling
        let half_width = width / 2.0;
        let half_height = height / 2.0;
        self.x1 = center_x - half_width;
        self.y1 = center_y - half_height;
        self.x2 = center_x + half_width;
        self.y2 = center_y + half_height;

        // Clip to boundaries if max_dims is provided
        if let Some((max_x, max_y)) = max_xy {
            self.x1 = self.x1.max(0.0);
            self.y1 = self.y1.max(0.0);
            self.x2 = self.x2.min(max_x);
            self.y2 = self.y2.min(max_y);
        }
    }
}

impl BboxXYWH {
    fn scale_uniformly(&mut self, factor: f32, max_xy: Option<(f32, f32)>) {
        let mut xyxy: BboxXYXY = (*self).into();
        xyxy.scale_uniformly(factor, max_xy);
        *self = xyxy.into();
    }
}

impl From<BboxXYXY> for BboxXYWH {
    fn from(item: BboxXYXY) -> Self {
        let center_x = (item.x1 + item.x2) / 2.0;
        let center_y = (item.y1 + item.y2) / 2.0;
        let width = (item.x2 - item.x1).abs();
        let height = (item.y2 - item.y1).abs();
        Self {
            center_x,
            center_y,
            width,
            height,
        }
    }
}

impl From<BboxXYWH> for BboxXYXY {
    fn from(item: BboxXYWH) -> Self {
        let half_width = item.width / 2.0;
        let half_height = item.height / 2.0;
        Self {
            x1: item.center_x - half_width,
            y1: item.center_y - half_height,
            x2: item.center_x + half_width,
            y2: item.center_y + half_height,
        }
    }
}

impl Bbox {
    pub fn to_xyxy(&self) -> BboxXYXY {
        match self {
            Bbox::BboxXYXY(bbox) => *bbox,
            Bbox::BboxXYWH(bbox) => (*bbox).into(),
        }
    }

    pub fn to_xywh(&self) -> BboxXYWH {
        match self {
            Bbox::BboxXYWH(bbox) => *bbox,
            Bbox::BboxXYXY(bbox) => (*bbox).into(),
        }
    }

    pub fn calculate_iou(&self, other: &Bbox) -> f32 {
        let det1 = self.to_xyxy();
        let det2 = other.to_xyxy();

        let x1_1 = det1.x1;
        let y1_1 = det1.y1;
        let x2_1 = det1.x2;
        let y2_1 = det1.y2;
        let x1_2 = det2.x1;
        let y1_2 = det2.y1;
        let x2_2 = det2.x2;
        let y2_2 = det2.y2;

        // Calculate intersection area
        let inter_x1 = x1_1.max(x1_2);
        let inter_y1 = y1_1.max(y1_2);
        let inter_x2 = x2_1.min(x2_2);
        let inter_y2 = y2_1.min(y2_2);
        let inter_area = if inter_x2 > inter_x1 && inter_y2 > inter_y1 {
            (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        } else {
            0.0
        };

        // Calculate union area
        let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        let union_area = area1 + area2 - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    pub fn scale_uniformly(&mut self, factor: f32, max_dims: Option<(f32, f32)>) {
        match self {
            Bbox::BboxXYWH(bbox) => {
                bbox.scale_uniformly(factor, max_dims);
            }
            Bbox::BboxXYXY(bbox) => {
                bbox.scale_uniformly(factor, max_dims);
            }
        }
    }

    pub fn reverse_padding_and_scaling(&mut self, padding: (u32, u32), scale: f32) {
        match self {
            Bbox::BboxXYXY(bbox) => {
                bbox.reverse_padding_and_scaling(padding, scale);
            }
            Bbox::BboxXYWH(bbox) => {
                // Convert to XYXY, reverse padding and scale, then convert back
                let mut xyxy: BboxXYXY = (*bbox).into();
                xyxy.reverse_padding_and_scaling(padding, scale);
                *bbox = xyxy.into();
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Landmark2D {
    pub x: f32,
    pub y: f32,
}

impl Landmark2D {
    pub fn reverse_padding_and_scaling(&mut self, padding: (u32, u32), scale: f32) {
        self.x = (self.x - padding.0 as f32) / scale;
        self.y = (self.y - padding.1 as f32) / scale;
    }

    pub fn distance(&self, other: &Landmark2D) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

pub trait DetectorTrait {
    fn new(model_path: impl Into<PathBuf>) -> Result<Self>
    where
        Self: Sized;
    fn infer(&mut self, frame: &DynamicImage) -> Result<DetectorOutput>;
}

// From estimator.rs

const ROI_EXPANSION_FACTOR: f32 = 1.25;

#[derive(Clone, Debug)]
pub enum ModelSize {
    Nano,
    Lite,
    Medium,
    Full,
    Large,
    Heavy,
}

impl Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSize::Nano => write!(f, "Nano"),
            ModelSize::Lite => write!(f, "Lite"),
            ModelSize::Medium => write!(f, "Medium"),
            ModelSize::Full => write!(f, "Full"),
            ModelSize::Large => write!(f, "Large"),
            ModelSize::Heavy => write!(f, "Heavy"),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PoseResult {
    pub pixel_landmarks: Vec<Landmark3D>,
    pub average_confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Landmark3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
    pub visibility: Option<f32>,
}

impl From<Landmark3D> for Landmark2D {
    fn from(landmark_3d: Landmark3D) -> Self {
        Landmark2D {
            x: landmark_3d.x,
            y: landmark_3d.y,
        }
    }
}

impl Landmark3D {
    pub fn new(x: f32, y: f32, z: f32, confidence: f32, visibility: Option<f32>) -> Self {
        Self {
            x,
            y,
            z,
            confidence,
            visibility,
        }
    }
}

impl RegionOfInterest {
    pub fn from_pose_result(
        pose_result: &PoseResult,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<Self> {
        // Default case: use an expanded tight bounding box
        // to calculate the bounding box for the next frame
        let frame_width = frame_width as f32;
        let frame_height = frame_height as f32;
        // Extract valid landmarks
        let valid_coords: Vec<(f32, f32)> = pose_result
            .pixel_landmarks
            .iter()
            .filter(|l| l.x.is_finite() && l.y.is_finite() && l.confidence >= 0.1)
            .map(|l| (l.x.clamp(0.0, frame_width), l.y.clamp(0.0, frame_height)))
            .collect();

        let mut bbox = if valid_coords.is_empty() {
            // No valid landmarks, use full frame
            Bbox::BboxXYXY(BboxXYXY {
                x1: 0.0,
                y1: 0.0,
                x2: frame_width,
                y2: frame_height,
            })
        } else {
            let x1 = valid_coords
                .iter()
                .map(|(x, _)| x)
                .fold(f32::INFINITY, |a, &b| a.min(b));
            let x2 = valid_coords
                .iter()
                .map(|(x, _)| x)
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let y1 = valid_coords
                .iter()
                .map(|(_, y)| y)
                .fold(f32::INFINITY, |a, &b| a.min(b));
            let y2 = valid_coords
                .iter()
                .map(|(_, y)| y)
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            Bbox::BboxXYXY(BboxXYXY { x1, y1, x2, y2 })
        };
        bbox.scale_uniformly(ROI_EXPANSION_FACTOR, Some((frame_width, frame_height)));

        Ok(Self {
            bbox,
            confidence: 1.0,
            landmarks: None,
            rotation: None,
        })
    }
}

pub trait EstimatorTrait {
    fn new(model_type: ModelSize, model_path: impl Into<PathBuf>) -> Result<Self>
    where
        Self: Sized;

    fn infer(&mut self, frame: &DynamicImage, roi: &RegionOfInterest)
        -> Result<Option<PoseResult>>;

    fn to_model_name(model_size: ModelSize) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Bbox tests (from detector.rs)
    #[test]
    fn test_bboxxyxy_scale_uniformly() {
        let bbox_xyxy = BboxXYXY {
            x1: 100.0,
            y1: 50.0,
            x2: 200.0,
            y2: 150.0,
        };
        let mut bbox_xywh: BboxXYWH = bbox_xyxy.into();
        bbox_xywh.scale_uniformly(1.25, None);
        assert_abs_diff_eq!(bbox_xywh.center_x, 150.0, epsilon = 0.001);
        assert_abs_diff_eq!(bbox_xywh.center_y, 100.0, epsilon = 0.001);
        assert_abs_diff_eq!(bbox_xywh.width, 125.0, epsilon = 0.001);
        assert_abs_diff_eq!(bbox_xywh.height, 125.0, epsilon = 0.001);
    }

    #[test]
    fn test_bbox_scale_uniformly_different_aspect_ratio() {
        let bbox_xyxy = BboxXYXY {
            x1: 10.0,
            y1: 20.0,
            x2: 50.0,
            y2: 80.0,
        };
        let factor = 1.0;
        let mut bbox_xywh: BboxXYWH = bbox_xyxy.into();
        bbox_xywh.scale_uniformly(factor, None);
        assert_abs_diff_eq!(bbox_xywh.center_x, 30.0, epsilon = 0.001);
        assert_abs_diff_eq!(bbox_xywh.center_y, 50.0, epsilon = 0.001);
        assert_abs_diff_eq!(bbox_xywh.width, 40.0, epsilon = 0.001);
        assert_abs_diff_eq!(bbox_xywh.height, 60.0, epsilon = 0.001);
    }

    #[test]
    fn test_iou_same_format() {
        let bbox1 = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        });
        let bbox2 = Bbox::BboxXYXY(BboxXYXY {
            x1: 5.0,
            y1: 5.0,
            x2: 15.0,
            y2: 15.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert!((iou - 0.1428).abs() < 0.001);
    }

    #[test]
    fn test_iou_different_formats() {
        let bbox1 = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        });
        let bbox2 = Bbox::BboxXYWH(BboxXYWH {
            center_x: 10.0,
            center_y: 10.0,
            width: 10.0,
            height: 10.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert!((iou - 0.1428).abs() < 0.001);
    }

    #[test]
    fn test_iou_no_overlap() {
        let bbox1 = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        });
        let bbox2 = Bbox::BboxXYXY(BboxXYXY {
            x1: 20.0,
            y1: 20.0,
            x2: 30.0,
            y2: 30.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_iou_identical() {
        let bbox1 = Bbox::BboxXYWH(BboxXYWH {
            center_x: 5.0,
            center_y: 5.0,
            width: 10.0,
            height: 10.0,
        });
        let bbox2 = Bbox::BboxXYWH(BboxXYWH {
            center_x: 5.0,
            center_y: 5.0,
            width: 10.0,
            height: 10.0,
        });
        let iou = bbox1.calculate_iou(&bbox2);
        assert_eq!(iou, 1.0);
    }

    #[test]
    fn test_scale_uniformly_xywh() {
        let mut bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 10.0,
            center_y: 10.0,
            width: 20.0,
            height: 10.0,
        });
        bbox.scale_uniformly(2.0, None);
        if let Bbox::BboxXYWH(xywh) = bbox {
            assert_eq!(xywh.center_x, 10.0);
            assert_eq!(xywh.center_y, 10.0);
            assert_eq!(xywh.width, 40.0);
            assert_eq!(xywh.height, 20.0);
        } else {
            panic!("Expected BboxXYWH");
        }
    }

    #[test]
    fn test_scale_uniformly_xyxy() {
        let mut bbox = Bbox::BboxXYXY(BboxXYXY {
            x1: 0.0,
            y1: 5.0,
            x2: 20.0,
            y2: 15.0,
        });
        bbox.scale_uniformly(2.0, None);
        if let Bbox::BboxXYXY(xyxy) = bbox {
            assert_eq!(xyxy.x1, -10.0);
            assert_eq!(xyxy.y1, 0.0);
            assert_eq!(xyxy.x2, 30.0);
            assert_eq!(xyxy.y2, 20.0);
        } else {
            panic!("Expected BboxXYXY");
        }
    }

    #[test]
    fn test_scale_uniformly_shrink() {
        let mut bbox = Bbox::BboxXYXY(BboxXYXY {
            x1: 10.0,
            y1: 10.0,
            x2: 30.0,
            y2: 30.0,
        });
        bbox.scale_uniformly(0.5, None);
        if let Bbox::BboxXYXY(xyxy) = bbox {
            assert_eq!(xyxy.x1, 15.0);
            assert_eq!(xyxy.y1, 15.0);
            assert_eq!(xyxy.x2, 25.0);
            assert_eq!(xyxy.y2, 25.0);
        } else {
            panic!("Expected BboxXYXY");
        }
    }

    #[test]
    fn test_roi_reverse_padding_and_scaling() {
        let mut roi = RegionOfInterest {
            bbox: Bbox::BboxXYXY(BboxXYXY {
                x1: 20.0,
                y1: 20.0,
                x2: 20.0,
                y2: 20.0,
            }),
            confidence: 0.9,
            landmarks: Some(vec![Landmark2D { x: 20.0, y: 20.0 }]),
            rotation: None,
        };
        let padding = (10, 10);
        let scaling = 2.0;
        let expected_coord = 5.0;
        roi.reverse_padding_and_scaling(padding, scaling);
        assert_eq!(
            roi.bbox,
            Bbox::BboxXYXY(BboxXYXY {
                x1: expected_coord,
                y1: expected_coord,
                x2: expected_coord,
                y2: expected_coord
            })
        );
        assert_eq!(roi.confidence, 0.9);
        assert_eq!(
            roi.landmarks,
            Some(vec![Landmark2D {
                x: expected_coord,
                y: expected_coord
            }])
        );
    }

    #[test]
    fn test_landmark_distance() {
        let kp1 = Landmark2D { x: 10.0, y: 10.0 };
        let kp2 = Landmark2D { x: 13.0, y: 14.0 };
        let expected_distance = 5.0;
        let distance = kp1.distance(&kp2);
        assert_eq!(distance, expected_distance);
    }

    // Estimator type tests (from estimator.rs)
    #[test]
    fn test_new_landmark3d() {
        let landmark = Landmark3D::new(1.0, 2.0, 3.0, 0.9, None);
        assert_eq!(landmark.x, 1.0);
        assert_eq!(landmark.y, 2.0);
        assert_eq!(landmark.z, 3.0);
        assert_eq!(landmark.confidence, 0.9);
        assert_eq!(landmark.visibility, None);
    }

    #[test]
    fn test_from_pose_result_uses_expanded_landmark_bounds() {
        let pixel_landmarks = vec![
            Landmark3D::new(50.0, 60.0, 0.0, 0.9, None),
            Landmark3D::new(150.0, 160.0, 0.0, 0.8, None),
            Landmark3D::new(100.0, 200.0, 0.0, 0.7, None),
        ];
        let pose_result = PoseResult {
            pixel_landmarks,
            average_confidence: 0.8,
        };
        let roi = RegionOfInterest::from_pose_result(&pose_result, 640, 480);
        let roi = roi.unwrap();
        assert_eq!(roi.confidence, 1.0);
        assert!(roi.rotation.is_none());
        assert!(roi.landmarks.is_none());
        match roi.bbox {
            Bbox::BboxXYXY(bbox) => {
                assert_eq!(bbox.x1, 37.5);
                assert_eq!(bbox.x2, 162.5);
                assert_eq!(bbox.y1, 42.5);
                assert_eq!(bbox.y2, 217.5);
            }
            _ => panic!("Expected BboxXYXY"),
        }
    }

    #[test]
    fn test_from_pose_result_no_finite_landmarks() {
        let pixel_landmarks = vec![
            Landmark3D::new(f32::NAN, f32::NAN, 0.0, 0.9, None),
            Landmark3D::new(f32::INFINITY, f32::INFINITY, 0.0, 0.8, None),
            Landmark3D::new(f32::NEG_INFINITY, 50.0, 0.0, 0.7, None),
        ];
        let pose_result = PoseResult {
            pixel_landmarks,
            average_confidence: 0.8,
        };
        let roi = RegionOfInterest::from_pose_result(&pose_result, 640, 480);
        let roi = roi.unwrap();
        match roi.bbox {
            Bbox::BboxXYXY(bbox) => {
                assert_eq!(bbox.x1, 0.0);
                assert_eq!(bbox.x2, 640.0);
                assert_eq!(bbox.y1, 0.0);
                assert_eq!(bbox.y2, 480.0);
            }
            _ => panic!("Expected BboxXYXY"),
        }
    }

    use parameterized::parameterized;

    #[parameterized(
        model_size = {ModelSize::Nano, ModelSize::Lite, ModelSize::Medium, ModelSize::Full, ModelSize::Large, ModelSize::Heavy},
        expected_path = {"Nano", "Lite", "Medium", "Full", "Large", "Heavy"}
    )]
    fn test_model_size_display(model_size: ModelSize, expected_path: &str) {
        assert_eq!(model_size.to_string(), expected_path);
    }
}
