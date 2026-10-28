use super::config::{
    FACE_RANGE, FOOT_EDGES, HAND_EDGES_RELATIVE, LEFT_HAND_RANGE, RIGHT_HAND_RANGE,
    RTMW_LANDMARK_CONFIDENCE_THRESHOLD,
};
use crate::config::COCO_SKELETON_EDGES;
use crate::shared::colors::{RUST, TURQUOISE, WHITE};
use crate::shared::types::{PoseResult, RegionOfInterest};
use crate::shared::visualization::{
    draw_landmark_dot, draw_skeleton_line, draw_thick_bbox, skeleton_dot_radius,
};
use crate::shared::visualization::{BBOX_COLOR, BBOX_THICKNESS};
use anyhow::Result;
use image::{DynamicImage, GenericImageView, RgbImage};

const BONE_COLOR: [u8; 3] = RUST;
const FACE_DOT_COLOR: [u8; 3] = TURQUOISE;

fn draw_edge(img: &mut RgbImage, pose: &PoseResult, a: usize, b: usize, color: [u8; 3]) {
    if let (Some(la), Some(lb)) = (pose.pixel_landmarks.get(a), pose.pixel_landmarks.get(b)) {
        if la.confidence >= RTMW_LANDMARK_CONFIDENCE_THRESHOLD
            && lb.confidence >= RTMW_LANDMARK_CONFIDENCE_THRESHOLD
        {
            draw_skeleton_line(img, la.x, la.y, lb.x, lb.y, color);
        }
    }
}

pub fn draw_rtmw_predictions_on_image(
    frame: &DynamicImage,
    pose_results: &[Option<PoseResult>],
    detections: Option<&[RegionOfInterest]>,
    landmarks_size: Option<f32>,
) -> Result<RgbImage> {
    let mut img = frame.to_rgb8();
    let (width, height) = frame.dimensions();
    let dot_radius = landmarks_size
        .map(|s| s.round() as i32)
        .unwrap_or_else(|| skeleton_dot_radius(width, height));
    // Face landmarks are tiny and dense; a body-sized dot smears them into a
    // blob. One pixel keeps the 68 face points readable regardless of the
    // image-size-driven body radius.
    let face_dot_radius: i32 = 1;

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
        // Body keypoints follow COCO-17 ordering exactly, so the body skeleton
        // table from the body-only pipeline applies verbatim.
        for &(a, b) in &COCO_SKELETON_EDGES {
            draw_edge(&mut img, pose, a, b, BONE_COLOR);
        }
        for &(a, b) in &FOOT_EDGES {
            draw_edge(&mut img, pose, a, b, BONE_COLOR);
        }
        // Both hands share the same relative edge table; we apply it twice
        // with the per-side base offset.
        for hand_base in [LEFT_HAND_RANGE.start, RIGHT_HAND_RANGE.start] {
            for &(ra, rb) in &HAND_EDGES_RELATIVE {
                draw_edge(&mut img, pose, hand_base + ra, hand_base + rb, BONE_COLOR);
            }
        }

        for (i, lm) in pose.pixel_landmarks.iter().enumerate() {
            if lm.confidence < RTMW_LANDMARK_CONFIDENCE_THRESHOLD {
                continue;
            }
            let in_face = FACE_RANGE.contains(&i);
            let color = if in_face { FACE_DOT_COLOR } else { WHITE };
            let r = if in_face { face_dot_radius } else { dot_radius };
            draw_landmark_dot(&mut img, lm.x, lm.y, r, color);
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::types::Landmark3D;
    use crate::wholebody::config::N_WHOLEBODY_KEYPOINTS;
    use image::RgbImage;

    fn fake_wholebody_pose() -> PoseResult {
        // 133 keypoints, spaced so we don't overlap dots.
        let lms: Vec<Landmark3D> = (0..N_WHOLEBODY_KEYPOINTS)
            .map(|i| {
                let x = 20.0 + (i % 13) as f32 * 4.0;
                let y = 20.0 + (i / 13) as f32 * 4.0;
                Landmark3D::new(x, y, 0.0, 0.9, Some(0.9))
            })
            .collect();
        PoseResult {
            pixel_landmarks: lms,
            average_confidence: 0.9,
        }
    }

    #[test]
    fn test_draw_wholebody_runs_without_panicking() {
        // Arrange
        let img = DynamicImage::ImageRgb8(RgbImage::new(200, 200));
        let poses = vec![Some(fake_wholebody_pose())];

        // Act
        let result = draw_rtmw_predictions_on_image(&img, &poses, None, None);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimensions(), (200, 200));
    }

    #[test]
    fn test_draw_wholebody_skips_low_confidence_landmarks() {
        // Arrange — every landmark below threshold; the body bone colour must
        // never appear in the rendered image.
        let mut pose = fake_wholebody_pose();
        for lm in pose.pixel_landmarks.iter_mut() {
            lm.confidence = 0.1;
        }
        let img = DynamicImage::ImageRgb8(RgbImage::new(96, 96));

        // Act
        let out = draw_rtmw_predictions_on_image(&img, &[Some(pose)], None, None).unwrap();

        // Assert
        let any_bone = out.pixels().any(|px| px.0 == BONE_COLOR);
        assert!(
            !any_bone,
            "below-threshold landmarks must not produce any bone-coloured pixels"
        );
    }
}
