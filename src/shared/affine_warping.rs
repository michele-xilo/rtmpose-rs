use super::image::RGB_CHANNELS;
use super::types::BorderMode;
use crate::shared::types::RegionOfInterest;
use nalgebra::{Matrix3, Vector3};
use std::f32;

pub fn warp_affine(
    src_image: &[u8],
    src_dimensions: (usize, usize),
    transform_matrix: &Matrix3<f32>,
    dst_dimensions: (usize, usize),
    border_mode: BorderMode,
) -> Vec<u8> {
    let dst_width = dst_dimensions.0;
    let dst_height = dst_dimensions.1;
    let src_width = src_dimensions.0;
    let src_height = src_dimensions.1;

    let mut dst = vec![0u8; dst_width * dst_height * RGB_CHANNELS];

    if let Some(inv_matrix) = transform_matrix.try_inverse() {
        for dst_y in 0..dst_height {
            for dst_x in 0..dst_width {
                // Apply inverse transformation to get source coordinates
                let dst_point = Vector3::new(dst_x as f32, dst_y as f32, 1.0);
                let src_point = inv_matrix * dst_point;

                let src_x = src_point.x;
                let src_y = src_point.y;

                // Bilinear interpolation
                let sx0 = src_x.floor() as i32;
                let sy0 = src_y.floor() as i32;
                let sx1 = sx0 + 1;
                let sy1 = sy0 + 1;

                let fx = src_x - sx0 as f32;
                let fy = src_y - sy0 as f32;

                for c in 0..RGB_CHANNELS {
                    let p00 =
                        get_pixel_value(src_image, src_width, src_height, sx0, sy0, c, border_mode)
                            as f32;
                    let p01 =
                        get_pixel_value(src_image, src_width, src_height, sx1, sy0, c, border_mode)
                            as f32;
                    let p10 =
                        get_pixel_value(src_image, src_width, src_height, sx0, sy1, c, border_mode)
                            as f32;
                    let p11 =
                        get_pixel_value(src_image, src_width, src_height, sx1, sy1, c, border_mode)
                            as f32;

                    let value = (1.0 - fx) * (1.0 - fy) * p00
                        + fx * (1.0 - fy) * p01
                        + (1.0 - fx) * fy * p10
                        + fx * fy * p11;

                    dst[(dst_y * dst_width + dst_x) * RGB_CHANNELS + c] = value.round() as u8;
                }
            }
        }
    } else {
        panic!("The transform matrix is not invertable.")
    }

    dst
}

fn get_pixel_value(
    src: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    channel: usize,
    border_mode: BorderMode,
) -> u8 {
    let (x, y) = match border_mode {
        BorderMode::Constant(value) => {
            if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 {
                return value;
            }
            (x as usize, y as usize)
        }
        BorderMode::Replicate => {
            let x = x.clamp(0, width as i32 - 1) as usize;
            let y = y.clamp(0, height as i32 - 1) as usize;
            (x, y)
        }
    };

    src[(y * width + x) * RGB_CHANNELS + channel]
}

pub fn get_affine_transform_matrix(
    roi: &RegionOfInterest,
    target_dimensions: (u32, u32),
) -> (Matrix3<f32>, Matrix3<f32>) {
    let bbox_xywh = roi.bbox.to_xywh();

    // Fix aspect ratio to match target dimensions
    let target_aspect = target_dimensions.0 as f32 / target_dimensions.1 as f32;
    let bbox_aspect = bbox_xywh.width / bbox_xywh.height;

    let (scale_w, scale_h) = if bbox_aspect > target_aspect {
        // Fit width
        let scale = target_dimensions.0 as f32 / bbox_xywh.width;
        (scale, scale)
    } else {
        // Fit height
        let scale = target_dimensions.1 as f32 / bbox_xywh.height;
        (scale, scale)
    };

    // 1. Translate ROI center to origin - t1
    // 2. Rotate - r
    // 3. Scale - s
    // 4. Translate to target center - t2
    let t1 = Matrix3::new(
        1.0,
        0.0,
        -bbox_xywh.center_x,
        0.0,
        1.0,
        -bbox_xywh.center_y,
        0.0,
        0.0,
        1.0,
    );

    let mut r = Matrix3::identity();
    if let Some(rotation) = roi.rotation {
        let cos_theta = rotation.cos();
        let sin_theta = rotation.sin();
        r = Matrix3::new(
            cos_theta, -sin_theta, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 1.0,
        );
    };

    let s = Matrix3::new(scale_w, 0.0, 0.0, 0.0, scale_h, 0.0, 0.0, 0.0, 1.0);

    let t2 = Matrix3::new(
        1.0,
        0.0,
        target_dimensions.0 as f32 * 0.5,
        0.0,
        1.0,
        target_dimensions.1 as f32 * 0.5,
        0.0,
        0.0,
        1.0,
    );

    (t2 * s * r * t1, r)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::shared::types::{Bbox, BboxXYWH};
    use approx::assert_abs_diff_eq;
    use image::DynamicImage;

    // A gradient instead of a photo: these tests only check geometry, and the
    // repo ships no images.
    fn synthetic_image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(320, 240, |x, y| {
            image::Rgb([x as u8, y as u8, (x + y) as u8])
        }))
    }

    use parameterized::parameterized;

    #[parameterized(border_mode = { BorderMode::Constant(0), BorderMode::Replicate })]
    fn test_warp_affine_with_rotation(border_mode: BorderMode) {
        // Arrange
        let image = synthetic_image();
        let rgb_img = image.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let src_image = rgb_img.as_raw();

        // Create a rotation matrix (45 degrees around center)
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        let angle = std::f32::consts::PI / 4.0; // 45 degrees in radians
        let scale = 1.0;

        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let alpha = scale * cos_a;
        let beta = scale * sin_a;

        let transform_matrix = Matrix3::new(
            alpha,
            beta,
            (1.0 - alpha) * center_x - beta * center_y,
            -beta,
            alpha,
            beta * center_x + (1.0 - alpha) * center_y,
            0.0,
            0.0,
            1.0,
        );

        // Act - Apply the affine transformation
        let transformed_data = warp_affine(
            src_image,
            (width as usize, height as usize),
            &transform_matrix,
            (width as usize, height as usize),
            border_mode,
        );

        // Assert
        let transformed_img =
            image::ImageBuffer::from_raw(width, height, transformed_data).unwrap();

        let output_image = DynamicImage::ImageRgb8(transformed_img);

        // Verify the output has the same dimensions as input
        assert_eq!(output_image.width(), width);
        assert_eq!(output_image.height(), height);
    }

    #[test]
    fn test_warp_affine_with_scaling() {
        // Arrange
        let image = synthetic_image();
        let rgb_img = image.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let src_image = rgb_img.as_raw();

        // Create a scaling matrix (scale by 0.8)
        let scale_x = 0.8;
        let scale_y = 0.8;
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        let transform_matrix = Matrix3::new(
            scale_x,
            0.0,
            (1.0 - scale_x) * center_x,
            0.0,
            scale_y,
            (1.0 - scale_y) * center_y,
            0.0,
            0.0,
            1.0,
        );

        // Act - Apply the affine transformation
        let transformed_data = warp_affine(
            src_image,
            (width as usize, height as usize),
            &transform_matrix,
            (width as usize, height as usize),
            BorderMode::Constant(114), // Use gray padding like in other tests
        );

        // Assert
        let transformed_img =
            image::ImageBuffer::from_raw(width, height, transformed_data).unwrap();

        let output_image = DynamicImage::ImageRgb8(transformed_img);

        // Verify the output has the same dimensions as input
        assert_eq!(output_image.width(), width);
        assert_eq!(output_image.height(), height);
    }

    #[test]
    fn test_warp_affine_with_translation() {
        // Arrange
        let image = synthetic_image();
        let rgb_img = image.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let src_image = rgb_img.as_raw();

        // Create a translation matrix (move right by 50 pixels, down by 30 pixels)
        let tx = 50.0;
        let ty = 30.0;

        let transform_matrix = Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0);

        // Act - Apply the affine transformation
        let transformed_data = warp_affine(
            src_image,
            (width as usize, height as usize),
            &transform_matrix,
            (width as usize, height as usize),
            BorderMode::Replicate,
        );

        // Assert
        let transformed_img =
            image::ImageBuffer::from_raw(width, height, transformed_data).unwrap();

        let output_image = DynamicImage::ImageRgb8(transformed_img);

        // Verify the output has the same dimensions as input
        assert_eq!(output_image.width(), width);
        assert_eq!(output_image.height(), height);
    }

    #[test]
    fn test_warp_affine_with_rotation_from_region_of_interest() {
        // Arrange
        let image = synthetic_image();
        let rgb_img = image.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let src_image = rgb_img.as_raw();
        let border_mode = BorderMode::Replicate;

        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 50.0,
            center_y: 50.0,
            width: 40.0,
            height: 40.0,
        });
        let target_dimensions = (80, 80);
        let rotation_angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let roi = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: Some(rotation_angle),
        };

        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(&roi, target_dimensions);

        // Act - Apply the affine transformation
        let transformed_data = warp_affine(
            src_image,
            (width as usize, height as usize),
            &transform_matrix,
            (target_dimensions.0 as usize, target_dimensions.1 as usize),
            border_mode,
        );

        // Assert
        let transformed_img = image::ImageBuffer::from_raw(
            target_dimensions.0,
            target_dimensions.1,
            transformed_data,
        )
        .unwrap();

        let output_image = DynamicImage::ImageRgb8(transformed_img);

        // Verify the output has the same dimensions as input
        assert_eq!(output_image.width(), target_dimensions.0);
        assert_eq!(output_image.height(), target_dimensions.1);
    }

    #[parameterized(coord = { (0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 2) }, expected = {
        12, 8, 4, 1
    })]
    fn test_get_pixel_value_coordinates_within_src(coord: (i32, i32, usize), expected: u8) {
        // Arrange
        let src = vec![
            12, 11, 10, // 0, 0, RGB
            9, 8, 7, // 1, 0, RGB
            6, 5, 4, // 0, 1, RGB
            3, 2, 1, // 1, 1, RGB
        ]; //RGBRGBRGB...
        let width = 2;
        let height = 2;

        // Act
        // Assert
        assert_eq!(
            get_pixel_value(
                &src,
                width,
                height,
                coord.0,
                coord.1,
                coord.2,
                BorderMode::Constant(0)
            ),
            expected
        );
    }

    #[parameterized( border_mode = { BorderMode::Constant(0), BorderMode::Replicate}, expected = {
        0, 3
    })]
    fn test_get_pixel_value_coordinates_outside_src(border_mode: BorderMode, expected: u8) {
        // Arrange
        // 2x2x3(RGB) image
        let src = vec![
            12, 11, 10, // 0, 0, RGB
            9, 8, 7, // 1, 0, RGB
            6, 5, 4, // 0, 1, RGB
            3, 2, 1, // 1, 1, RGB
        ]; //RGBRGBRGB...
        let width = 2;
        let height = 2;
        let coord = (3, 3, 0); // will be clamped at 1,1,0 for BorderMode::Replicate

        // Act
        // Assert
        assert_eq!(
            get_pixel_value(&src, width, height, coord.0, coord.1, coord.2, border_mode),
            expected
        );
    }
    #[test]
    fn test_get_affine_transform() {
        // Arrange
        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 0.0,
            center_y: 0.0,
            width: 2.0,
            height: 2.0,
        });
        let target_dimensions = (4, 4);
        let roi = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: None,
        };

        // Act
        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(&roi, target_dimensions);

        // Assert
        // Expected: scale by 2x and translate by (2, 2) to center the bbox
        let expected_matrix = Matrix3::new(
            2.0, 0.0, 2.0, // scale_x=2, tx=2
            0.0, 2.0, 2.0, // scale_y=2, ty=2
            0.0, 0.0, 1.0, // homogeneous coordinates
        );

        assert_eq!(transform_matrix, expected_matrix);
    }

    #[test]
    fn test_get_affine_transform_different_aspect_ratios() {
        // Arrange - bbox is wider than target (fit width case)
        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 100.0,
            center_y: 200.0,
            width: 400.0,
            height: 200.0, // aspect ratio 2:1
        });
        let target_dimensions = (256, 256); // aspect ratio 1:1
        let roi = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: None,
        };
        // Act
        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(&roi, target_dimensions);

        // Assert
        // bbox_aspect (2.0) > target_aspect (1.0), so fit width
        // scale = 256 / 400 = 0.64
        let expected_scale = 0.64;
        let expected_tx = 256.0 * 0.5 - 100.0 * expected_scale; // 128 - 64 = 64
        let expected_ty = 256.0 * 0.5 - 200.0 * expected_scale; // 128 - 128 = 0

        let expected_matrix = Matrix3::new(
            expected_scale,
            0.0,
            expected_tx,
            0.0,
            expected_scale,
            expected_ty,
            0.0,
            0.0,
            1.0,
        );

        assert_eq!(transform_matrix, expected_matrix);
    }

    #[test]
    fn test_get_affine_transform_tall_bbox() {
        // Arrange - bbox is taller than target (fit height case)
        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 50.0,
            center_y: 75.0,
            width: 100.0,
            height: 300.0, // aspect ratio 1:3
        });
        let target_dimensions = (192, 256); // aspect ratio 3:4
        let roi = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: None,
        };
        // Act
        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(&roi, target_dimensions);

        // Assert
        // bbox_aspect (1/3 ≈ 0.33) < target_aspect (3/4 = 0.75), so fit height
        // scale = 256 / 300 ≈ 0.8533
        let expected_scale = 256.0 / 300.0;
        let expected_tx = 192.0 * 0.5 - 50.0 * expected_scale; // 96 - ~42.67 ≈ 53.33
        let expected_ty = 256.0 * 0.5 - 75.0 * expected_scale; // 128 - 64 = 64

        let expected_matrix = Matrix3::new(
            expected_scale,
            0.0,
            expected_tx,
            0.0,
            expected_scale,
            expected_ty,
            0.0,
            0.0,
            1.0,
        );

        // Use approximate equality for floating point comparison
        assert!((transform_matrix[(0, 0)] - expected_matrix[(0, 0)]).abs() < 1e-6);
        assert!((transform_matrix[(1, 1)] - expected_matrix[(1, 1)]).abs() < 1e-6);
        assert!((transform_matrix[(0, 2)] - expected_matrix[(0, 2)]).abs() < 1e-6);
        assert!((transform_matrix[(1, 2)] - expected_matrix[(1, 2)]).abs() < 1e-6);
        assert_eq!(transform_matrix[(2, 2)], 1.0);
    }

    #[test]
    fn test_get_affine_transform_with_90_degree_rotation() {
        // Arrange
        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 50.0,
            center_y: 50.0,
            width: 40.0,
            height: 40.0,
        });
        let target_dimensions = (80, 80);
        let rotation_angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let roi = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: Some(rotation_angle),
        };

        // Act
        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(&roi, target_dimensions);

        // Assert
        // For 90-degree rotation: cos(90°) = 0, sin(90°) = 1
        // Scale = 80 / 40 = 2.0
        let scale = 2.0;

        // Build expected matrix step by step
        let t1 = Matrix3::new(1.0, 0.0, -50.0, 0.0, 1.0, -50.0, 0.0, 0.0, 1.0);
        let r = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0); // 90-degree rotation
        let s = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
        let t2 = Matrix3::new(1.0, 0.0, 40.0, 0.0, 1.0, 40.0, 0.0, 0.0, 1.0);

        let expected_matrix = t2 * s * r * t1;

        // Verify the rotation components are correct for 90 degrees
        // The off-diagonal elements should show the rotation
        assert_abs_diff_eq!(
            transform_matrix[(0, 1)],
            expected_matrix[(0, 1)],
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            transform_matrix[(1, 0)],
            expected_matrix[(1, 0)],
            epsilon = 1e-3
        );

        // Verify all matrix elements
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    transform_matrix[(i, j)],
                    expected_matrix[(i, j)],
                    epsilon = 1e-3
                );
            }
        }
    }

    #[test]
    fn test_get_affine_transform_with_180_degree_rotation() {
        // Arrange
        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 100.0,
            center_y: 75.0,
            width: 60.0,
            height: 60.0,
        });
        let target_dimensions = (120, 120);
        let rotation_angle = std::f32::consts::PI; // 180 degrees
        let roi = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: Some(rotation_angle),
        };

        // Act
        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(&roi, target_dimensions);

        // Assert
        // For 180-degree rotation: cos(180°) = -1, sin(180°) = 0
        // Scale = 120 / 60 = 2.0
        let scale = 2.0;

        // Build expected matrix
        let t1 = Matrix3::new(1.0, 0.0, -100.0, 0.0, 1.0, -75.0, 0.0, 0.0, 1.0);
        let r = Matrix3::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0); // 180-degree rotation
        let s = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
        let t2 = Matrix3::new(1.0, 0.0, 60.0, 0.0, 1.0, 60.0, 0.0, 0.0, 1.0);

        let expected_matrix = t2 * s * r * t1;

        // For 180-degree rotation, the diagonal should be negative scale
        // and off-diagonal should be near zero
        assert!(
            transform_matrix[(0, 1)].abs() < 1e-3,
            "Off-diagonal should be ~0 for 180° rotation"
        );
        assert!(
            transform_matrix[(1, 0)].abs() < 1e-3,
            "Off-diagonal should be ~0 for 180° rotation"
        );

        // Verify all matrix elements
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    transform_matrix[(i, j)],
                    expected_matrix[(i, j)],
                    epsilon = 1e-3
                );
            }
        }
    }

    #[test]
    fn test_get_affine_transform_with_zero_rotation() {
        // Arrange
        let bbox = Bbox::BboxXYWH(BboxXYWH {
            center_x: 25.0,
            center_y: 25.0,
            width: 50.0,
            height: 50.0,
        });
        let target_dimensions = (100, 100);
        let roi_with_zero_rotation = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: Some(0.0), // Explicitly set rotation to 0
        };
        let roi_without_rotation = RegionOfInterest {
            bbox,
            confidence: 0.9,
            landmarks: None,
            rotation: None, // No rotation
        };

        // Act
        let (transform_with_zero_rotation, _rotation_matrix) =
            get_affine_transform_matrix(&roi_with_zero_rotation, target_dimensions);
        let (transform_without_rotation, _rotation_matrix) =
            get_affine_transform_matrix(&roi_without_rotation, target_dimensions);

        // Assert
        // Both matrices should be equivalent
        // For 0-degree rotation: cos(0°) = 1, sin(0°) = 0 (identity rotation)
        // Scale = 100 / 50 = 2.0
        let scale = 2.0;

        // Build expected matrix (same as no rotation)
        let t1 = Matrix3::new(1.0, 0.0, -25.0, 0.0, 1.0, -25.0, 0.0, 0.0, 1.0);
        let r = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // Identity (0-degree rotation)
        let s = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
        let t2 = Matrix3::new(1.0, 0.0, 50.0, 0.0, 1.0, 50.0, 0.0, 0.0, 1.0);

        let expected_matrix = t2 * s * r * t1;

        // Verify that zero rotation produces the same result as no rotation
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (transform_with_zero_rotation[(i, j)] - transform_without_rotation[(i, j)])
                        .abs()
                        < 1e-6,
                    "Zero rotation should equal no rotation at ({}, {}): {} != {}",
                    i,
                    j,
                    transform_with_zero_rotation[(i, j)],
                    transform_without_rotation[(i, j)]
                );
                assert!(
                    (transform_with_zero_rotation[(i, j)] - expected_matrix[(i, j)]).abs() < 1e-6,
                    "Matrix element ({}, {}) differs from expected: {} != {}",
                    i,
                    j,
                    transform_with_zero_rotation[(i, j)],
                    expected_matrix[(i, j)]
                );
            }
        }

        // Verify off-diagonal elements are 0 (no rotation effect)
        assert!(
            transform_with_zero_rotation[(0, 1)].abs() < 1e-6,
            "No rotation in off-diagonal"
        );
        assert!(
            transform_with_zero_rotation[(1, 0)].abs() < 1e-6,
            "No rotation in off-diagonal"
        );
    }

    #[test]
    fn test_regression_rotation_direction_matches_mediapipe_sampling() {
        // Regression: for a hand pointing right, the ROI stores -pi/2 (negated
        // orientation) so the affine transform de-rotates the hand to upright.
        //
        // Fingertips at (center_x + d, center_y) should map to the TOP of the
        // output (fingers up).
        let roi = RegionOfInterest {
            bbox: Bbox::BboxXYWH(BboxXYWH {
                center_x: 100.0,
                center_y: 100.0,
                width: 80.0,
                height: 80.0,
            }),
            confidence: 0.9,
            landmarks: None,
            // Hand points right → orientation is pi/2 → ROI stores negated = -pi/2
            rotation: Some(-std::f32::consts::FRAC_PI_2),
        };
        let target = (224_u32, 224_u32);
        let (transform, _) = get_affine_transform_matrix(&roi, target);

        // Fingertip in image space: 40px to the right of center
        let fingertip_img = Vector3::new(140.0, 100.0, 1.0);
        let fingertip_out = transform * fingertip_img;

        // Fingertip should map to the top-center of the output (y < 112)
        assert_abs_diff_eq!(fingertip_out.x, 112.0, epsilon = 1.0);
        assert!(
            fingertip_out.y < 112.0,
            "Fingertip must map to top half (y={}, expected < 112). \
             If y > 112, the rotation direction is inverted.",
            fingertip_out.y
        );

        // Wrist in image space: 40px to the left of center
        let wrist_img = Vector3::new(60.0, 100.0, 1.0);
        let wrist_out = transform * wrist_img;

        // Wrist should map to the bottom-center of the output (y > 112)
        assert_abs_diff_eq!(wrist_out.x, 112.0, epsilon = 1.0);
        assert!(
            wrist_out.y > 112.0,
            "Wrist must map to bottom half (y={}, expected > 112)",
            wrist_out.y
        );
    }
}
