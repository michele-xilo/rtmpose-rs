use super::config::{COCO_SKELETON_EDGES, RTMPOSE_LANDMARK_CONFIDENCE_THRESHOLD};
use crate::shared::colors::{RUST, WHITE};
use crate::shared::types::{PoseResult, RegionOfInterest};
use crate::shared::visualization::{
    draw_landmark_dot, draw_skeleton_line, draw_thick_bbox, skeleton_dot_radius,
};
use crate::shared::visualization::{BBOX_COLOR, BBOX_THICKNESS};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, RgbImage};

const BONE_COLOR: [u8; 3] = RUST;

pub fn draw_rtm_predictions_on_image(
    frame: &DynamicImage,
    pose_results: &[Option<PoseResult>],
    detections: Option<&[RegionOfInterest]>,
    landmarks_size: Option<f32>,
) -> Result<RgbImage> {
    let mut img = frame.to_rgb8();
    let (width, height) = frame.dimensions();
    let dot_radius = landmarks_size
        .filter(|&s| s.is_finite() && s > 0.0)
        .map(|s| s.round() as i32)
        .unwrap_or_else(|| skeleton_dot_radius(width, height));

    if let Some(detections) = detections {
        for detection in detections {
            draw_thick_bbox(
                &mut img,
                &detection.bbox,
                BBOX_COLOR,
                BBOX_THICKNESS,
                width,
                height,
            );
        }
    }

    for pose in pose_results.iter().flatten() {
        let lms = &pose.pixel_landmarks;
        for &(a, b) in &COCO_SKELETON_EDGES {
            if let (Some(la), Some(lb)) = (lms.get(a), lms.get(b)) {
                if la.confidence >= RTMPOSE_LANDMARK_CONFIDENCE_THRESHOLD
                    && lb.confidence >= RTMPOSE_LANDMARK_CONFIDENCE_THRESHOLD
                {
                    draw_skeleton_line(&mut img, la.x, la.y, lb.x, lb.y, BONE_COLOR);
                }
            }
        }
        for lm in lms.iter() {
            if lm.confidence < RTMPOSE_LANDMARK_CONFIDENCE_THRESHOLD {
                continue;
            }
            draw_landmark_dot(&mut img, lm.x, lm.y, dot_radius, WHITE);
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::types::Landmark3D;
    use image::RgbImage;

    fn fake_pose() -> PoseResult {
        let lms: Vec<Landmark3D> = (0..17)
            .map(|i| Landmark3D::new(10.0 + i as f32, 20.0 + i as f32, 0.0, 0.9, Some(0.9)))
            .collect();
        PoseResult {
            pixel_landmarks: lms,
            average_confidence: 0.9,
        }
    }

    #[test]
    fn test_draw_runs_without_panicking_on_blank_image() {
        // Arrange
        let img = DynamicImage::ImageRgb8(RgbImage::new(200, 200));
        let poses = vec![Some(fake_pose())];

        // Act
        let result = draw_rtm_predictions_on_image(&img, &poses, None, None);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimensions(), (200, 200));
    }

    #[test]
    fn test_draw_skips_low_confidence_landmarks() {
        // Arrange — all landmarks below threshold; output should match the input
        // (no draws). We check by ensuring no pixel was modified to BONE_COLOR.
        let mut pose = fake_pose();
        for lm in pose.pixel_landmarks.iter_mut() {
            lm.confidence = 0.1;
        }
        let img = DynamicImage::ImageRgb8(RgbImage::new(64, 64));

        // Act
        let out = draw_rtm_predictions_on_image(&img, &[Some(pose)], None, None).unwrap();

        // Assert
        let mut any_bone_color = false;
        for px in out.pixels() {
            if px.0 == BONE_COLOR {
                any_bone_color = true;
                break;
            }
        }
        assert!(
            !any_bone_color,
            "low-confidence landmarks must not draw bones"
        );
    }
}
