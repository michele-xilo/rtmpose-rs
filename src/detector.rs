use super::config::{
    COCO_PERSON_CLASS_ID, IMAGENET_MEAN_0_255, IMAGENET_STD_0_255, RTMDET_INPUT_DIMENSIONS,
    RTMDET_INPUT_TENSOR_NAME, RTMDET_INPUT_TENSOR_SHAPE, RTMDET_NANO_MODEL_NAME,
    RTMDET_NMS_IOU_THRESHOLD, RTMDET_OUTPUT_DETS, RTMDET_OUTPUT_LABELS, RTMDET_SCORE_THRESHOLD,
};
use crate::execution::with_default_execution_providers;
use crate::shared::image::{
    get_resizing_scale, pad_image_to_target_size, raw_rgb_to_normalized_tensor, resize_image_fast,
};
use crate::shared::types::{
    Bbox, BboxXYXY, DetectorOutput, DetectorTrait, MeanStdNormalizationConfig, NormalizationType,
    RegionOfInterest,
};
use anyhow::Result;
use image::DynamicImage;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;
use tracing::error;

const RTMDET_PADDING_COLOR: [u8; 3] = [114, 114, 114];

pub struct RtmDetector {
    session: Session,
}

impl DetectorTrait for RtmDetector {
    fn new(model_dir: impl Into<PathBuf>) -> Result<Self> {
        let model_path = model_dir.into().join(RTMDET_NANO_MODEL_NAME);
        let session = with_default_execution_providers(Session::builder()?)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e: ort::Error<_>| ort::Error::<()>::from(e))?
            .commit_from_file(&model_path)?;
        Ok(Self { session })
    }

    fn infer(&mut self, frame: &DynamicImage) -> Result<DetectorOutput> {
        let frame_w = frame.width() as f32;
        let frame_h = frame.height() as f32;

        let scale = get_resizing_scale(frame, RTMDET_INPUT_DIMENSIONS);
        let mut letterboxed = resize_image_fast(frame, RTMDET_INPUT_DIMENSIONS);
        let padding = pad_image_to_target_size(
            &mut letterboxed,
            RTMDET_INPUT_DIMENSIONS,
            RTMDET_PADDING_COLOR,
        )?;

        let normalization = NormalizationType::MeanStd(MeanStdNormalizationConfig {
            mean: IMAGENET_MEAN_0_255,
            std: IMAGENET_STD_0_255,
        });
        let input_array = raw_rgb_to_normalized_tensor(
            letterboxed.to_rgb8().as_raw(),
            letterboxed.width(),
            letterboxed.height(),
            &normalization,
            &RTMDET_INPUT_TENSOR_SHAPE,
        )?;
        let input_tensor = Tensor::from_array(input_array)?;
        let outputs = self
            .session
            .run(ort::inputs![RTMDET_INPUT_TENSOR_NAME => input_tensor])?;

        let dets = outputs
            .get(RTMDET_OUTPUT_DETS)
            .ok_or_else(|| anyhow::anyhow!("RTMDet output `{}` missing", RTMDET_OUTPUT_DETS))?
            .try_extract_array::<f32>()?;
        let labels = outputs
            .get(RTMDET_OUTPUT_LABELS)
            .ok_or_else(|| anyhow::anyhow!("RTMDet output `{}` missing", RTMDET_OUTPUT_LABELS))?
            .try_extract_array::<i64>()?;

        // mmdeploy's static export shape is `[1, N, 5]` for dets and `[1, N]`
        // for labels. We tolerate the un-batched `[N, 5]` / `[N]` form too in
        // case the model was exported without the batch dim.
        let dets_shape = dets.shape();
        let (n, last) = match dets_shape.len() {
            3 => (dets_shape[1], dets_shape[2]),
            2 => (dets_shape[0], dets_shape[1]),
            other => {
                error!(
                    "RTMDet `dets` has unexpected rank {} (shape {:?})",
                    other, dets_shape
                );
                return Err(anyhow::anyhow!(
                    "Unexpected RTMDet dets shape: {:?}",
                    dets_shape
                ));
            }
        };
        if last != 5 {
            return Err(anyhow::anyhow!(
                "RTMDet `dets` last dim must be 5 (x1,y1,x2,y2,score), got {last}"
            ));
        }
        let dets_flat = dets
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("RTMDet dets tensor is not contiguous"))?;
        let labels_flat = labels
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("RTMDet labels tensor is not contiguous"))?;

        if labels_flat.len() < n || dets_flat.len() < n * 5 {
            return Err(anyhow::anyhow!(
                "RTMDet output length mismatch: labels={}, dets={}, expected n={} and n*5={}",
                labels_flat.len(),
                dets_flat.len(),
                n,
                n * 5
            ));
        }

        let mut rois: Vec<RegionOfInterest> = Vec::new();
        for i in 0..n {
            let row = &dets_flat[i * 5..(i + 1) * 5];
            let score = row[4];
            if !score.is_finite()
                || !row[0].is_finite()
                || !row[1].is_finite()
                || !row[2].is_finite()
                || !row[3].is_finite()
            {
                continue;
            }
            if score < RTMDET_SCORE_THRESHOLD {
                continue;
            }
            if labels_flat.get(i).copied() != Some(COCO_PERSON_CLASS_ID) {
                continue;
            }
            // Undo letterbox: subtract padding, divide by scale, then clamp to
            // the source frame so downstream affine warps don't fetch from
            // negative coordinates.
            let x1 = ((row[0] - padding.0 as f32) / scale).clamp(0.0, frame_w);
            let y1 = ((row[1] - padding.1 as f32) / scale).clamp(0.0, frame_h);
            let x2 = ((row[2] - padding.0 as f32) / scale).clamp(0.0, frame_w);
            let y2 = ((row[3] - padding.1 as f32) / scale).clamp(0.0, frame_h);
            if x2 <= x1 || y2 <= y1 {
                continue;
            }
            rois.push(RegionOfInterest {
                bbox: Bbox::BboxXYXY(BboxXYXY { x1, y1, x2, y2 }),
                confidence: score,
                landmarks: None,
                rotation: None,
            });
        }

        // mmdeploy bakes NMS into the export; we sort by confidence so the
        // tracker's max_n_poses cap keeps the strongest detections.
        rois.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Defensive second pass: in case the export ships without NMS, drop
        // boxes whose IoU exceeds the upstream threshold.
        let mut deduped: Vec<RegionOfInterest> = Vec::with_capacity(rois.len());
        for cand in rois {
            let overlaps = deduped
                .iter()
                .any(|kept| kept.bbox.calculate_iou(&cand.bbox) > RTMDET_NMS_IOU_THRESHOLD);
            if !overlaps {
                deduped.push(cand);
            }
        }

        Ok(DetectorOutput {
            regions_of_interest: deduped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_rtm_detector_construction() {
        // Arrange / Act
        let detector = RtmDetector::new("./models");

        // Assert
        assert!(
            detector.is_ok(),
            "RtmDetector construction must succeed when ONNX is present"
        );
    }

    #[test]
    #[ignore]
    fn test_rtm_detector_infer_returns_person_rois() {
        // Arrange
        // No photo ships with the repo, so point RTMPOSE_TEST_IMAGE at one with
        // at least one visible person.
        let path =
            std::env::var("RTMPOSE_TEST_IMAGE").unwrap_or_else(|_| "assets/test_image.jpg".into());
        let mut detector = RtmDetector::new("./models").unwrap();
        let image = crate::shared::image::load_image_from_path(&path).unwrap();

        // Act
        let result = detector.infer(&image).unwrap();

        // Assert
        assert!(
            !result.regions_of_interest.is_empty(),
            "expected at least one person ROI on the test image"
        );
        for roi in &result.regions_of_interest {
            assert!(roi.confidence >= RTMDET_SCORE_THRESHOLD);
        }
    }
}
