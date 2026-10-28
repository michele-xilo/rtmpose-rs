use super::types::NormalizationType;
use anyhow::{bail, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array4;

pub const RGB_CHANNELS: usize = 3;

pub fn load_image_from_path(path: &str) -> Result<DynamicImage> {
    let image = image::open(path)?;
    Ok(image)
}

pub fn resize_image(image: &DynamicImage, dim: (u32, u32)) -> DynamicImage {
    image.resize(dim.0, dim.1, FilterType::Triangle)
}

// Box-averaged downscale, aspect-preserving to fit within `dim`. ~3x faster
// than the Triangle resize above on 1080p sources, with
// comparable antialiasing quality for downscaling — the win comes from
// integer averaging instead of float convolution.
pub fn resize_image_fast(image: &DynamicImage, dim: (u32, u32)) -> DynamicImage {
    let scale = get_resizing_scale(image, dim);
    let tw = ((image.width() as f32 * scale) as u32).max(1);
    let th = ((image.height() as f32 * scale) as u32).max(1);
    DynamicImage::ImageRgb8(image::imageops::thumbnail(&image.to_rgb8(), tw, th))
}

// Non-aspect-preserving (stretch) counterpart to `resize_image_fast`, matching
// `DynamicImage::resize_exact` but via the fast box-average path.
pub fn resize_image_exact_fast(image: &DynamicImage, dim: (u32, u32)) -> DynamicImage {
    DynamicImage::ImageRgb8(image::imageops::thumbnail(&image.to_rgb8(), dim.0, dim.1))
}

pub fn get_resizing_scale(image: &DynamicImage, dim: (u32, u32)) -> f32 {
    let (orig_width, orig_height) = image.dimensions();
    let (target_width, target_height) = dim;

    // Calculate the scale factor that resize() will use
    let scale_x = target_width as f32 / orig_width as f32;
    let scale_y = target_height as f32 / orig_height as f32;

    scale_x.min(scale_y)
}

pub fn switch_br_channles(img: &DynamicImage) -> DynamicImage {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let mut bgr_img = ImageBuffer::new(width, height);

    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let rgb = pixel.0;
        let bgr_pixel = Rgb([rgb[2], rgb[1], rgb[0]]);
        bgr_img.put_pixel(x, y, bgr_pixel);
    }

    DynamicImage::ImageRgb8(bgr_img)
}

pub fn pad_image_to_target_size(
    img: &mut DynamicImage,
    dim: (u32, u32),
    color: [u8; 3],
) -> Result<(u32, u32)> {
    if img.width() == dim.0 && img.height() == dim.1 {
        return Ok((0, 0));
    }
    if img.width() > dim.0 || img.height() > dim.1 {
        bail!("Cannot pad image with dimensions smaller than image dimensions");
    }

    // Add padding so that it's exactly the desired size
    let padding_width = (dim.0 - img.width()) / 2;
    let padding_height = (dim.1 - img.height()) / 2;

    let mut padded_buffer =
        ImageBuffer::from_pixel(dim.0, dim.1, Rgb([color[0], color[1], color[2]]));

    let resized_rgb = img.to_rgb8();
    for (x, y, pixel) in resized_rgb.enumerate_pixels() {
        padded_buffer.put_pixel(x + padding_width, y + padding_height, *pixel);
    }

    *img = DynamicImage::ImageRgb8(padded_buffer);
    Ok((padding_width, padding_height))
}

pub fn raw_rgb_to_normalized_tensor(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    normalization_type: &NormalizationType,
    target_shape: &(usize, usize, usize, usize),
) -> Result<Array4<f32>> {
    if rgb_data.len() != (width * height * 3) as usize {
        bail!("RGB data length doesn't match width * height * 3");
    }

    let mut tensor = Array4::<f32>::zeros(*target_shape);
    // Determine layout from the channel-axis position. The RGB_CHANNELS == 3
    // match rules out ambiguous shapes like (1, 4, 4, 3) that a size-based
    // heuristic would silently misclassify.
    let is_nhwc = match *target_shape {
        (1, _, _, RGB_CHANNELS) => true,
        (1, RGB_CHANNELS, _, _) => false,
        _ => bail!(
            "Expected NHWC (1,H,W,3) or NCHW (1,3,H,W) tensor shape, got {:?}",
            target_shape
        ),
    };

    match normalization_type {
        NormalizationType::Linear => {
            // Format is NHWC for linear normalization (pose)
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let pixel_idx = (y * width as usize + x) * 3;
                    let r = (rgb_data[pixel_idx] as f32 * 2.0 / 255.0) - 1.0;
                    let g = (rgb_data[pixel_idx + 1] as f32 * 2.0 / 255.0) - 1.0;
                    let b = (rgb_data[pixel_idx + 2] as f32 * 2.0 / 255.0) - 1.0;
                    if is_nhwc {
                        tensor[[0, y, x, 0]] = r;
                        tensor[[0, y, x, 1]] = g;
                        tensor[[0, y, x, 2]] = b;
                    } else {
                        tensor[[0, 0, y, x]] = r;
                        tensor[[0, 1, y, x]] = g;
                        tensor[[0, 2, y, x]] = b;
                    }
                }
            }
        }
        NormalizationType::UnitRange => {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let pixel_idx = (y * width as usize + x) * 3;
                    let r = rgb_data[pixel_idx] as f32 / 255.0;
                    let g = rgb_data[pixel_idx + 1] as f32 / 255.0;
                    let b = rgb_data[pixel_idx + 2] as f32 / 255.0;
                    if is_nhwc {
                        tensor[[0, y, x, 0]] = r;
                        tensor[[0, y, x, 1]] = g;
                        tensor[[0, y, x, 2]] = b;
                    } else {
                        tensor[[0, 0, y, x]] = r;
                        tensor[[0, 1, y, x]] = g;
                        tensor[[0, 2, y, x]] = b;
                    }
                }
            }
        }
        NormalizationType::UnitScale => {
            // Format is NCHW, maps 0:255 to 0:1
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let pixel_idx = (y * width as usize + x) * 3;
                    tensor[[0, 0, y, x]] = rgb_data[pixel_idx] as f32 / 255.0;
                    tensor[[0, 1, y, x]] = rgb_data[pixel_idx + 1] as f32 / 255.0;
                    tensor[[0, 2, y, x]] = rgb_data[pixel_idx + 2] as f32 / 255.0;
                }
            }
        }
        NormalizationType::MeanStd(config) => {
            // Format is NCHW for mean/std normalization
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let pixel_idx = (y * width as usize + x) * 3;
                    tensor[[0, 0, y, x]] =
                        (rgb_data[pixel_idx] as f32 - config.mean[0]) / config.std[0];
                    tensor[[0, 1, y, x]] =
                        (rgb_data[pixel_idx + 1] as f32 - config.mean[1]) / config.std[1];
                    tensor[[0, 2, y, x]] =
                        (rgb_data[pixel_idx + 2] as f32 - config.mean[2]) / config.std[2];
                }
            }
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_RESIZE_DIMENSIONS: (u32, u32) = (640, 640);
    use approx::assert_abs_diff_eq;
    use image::GenericImageView;
    use tracing::info;

    // A gradient instead of a photo: these tests only check geometry, and the
    // repo ships no images.
    fn synthetic_image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(320, 240, |x, y| {
            image::Rgb([x as u8, y as u8, (x + y) as u8])
        }))
    }

    const TEST_PADDING_COLOR: [u8; 3] = [128, 128, 128];

    #[test]
    fn test_load_image_from_path_with_invalid_path() {
        // Act
        let result = load_image_from_path("invalid_path.jpg");

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_image() {
        // Arrange
        let image = synthetic_image();
        let aspect_ratio = image.width() as f32 / image.height() as f32;

        // Act
        let resized_image = resize_image(&image, TEST_RESIZE_DIMENSIONS);

        // Assert
        assert_eq!(resized_image.width(), TEST_RESIZE_DIMENSIONS.0);
        let resized_aspect_ratio = resized_image.width() as f32 / resized_image.height() as f32;
        assert_abs_diff_eq!(resized_aspect_ratio, aspect_ratio, epsilon = 0.1);
    }

    #[test]
    fn test_resize_image_fast_preserves_aspect_and_fits() {
        // Arrange — 1920x1080 into a 640 box
        let image = DynamicImage::new_rgb8(1920, 1080);

        // Act
        let resized = resize_image_fast(&image, (640, 640));

        // Assert — fits within the box, aspect preserved (16:9 -> 640x360)
        assert!(resized.width() <= 640 && resized.height() <= 640);
        assert_eq!((resized.width(), resized.height()), (640, 360));
    }

    #[test]
    fn test_resize_image_exact_fast_stretches_to_dims() {
        // Arrange — 1920x1080 stretched to a square, aspect NOT preserved
        let image = DynamicImage::new_rgb8(1920, 1080);

        // Act
        let resized = resize_image_exact_fast(&image, (384, 384));

        // Assert
        assert_eq!((resized.width(), resized.height()), (384, 384));
    }

    #[test]
    fn test_pad_image_happy_path() {
        // Arrange
        let image = synthetic_image();
        let mut image = resize_image(&image, TEST_RESIZE_DIMENSIONS);
        info!("Resizing image dimensions are {:?}", TEST_RESIZE_DIMENSIONS);
        assert_eq!(image.width(), TEST_RESIZE_DIMENSIONS.0);

        // Act
        let _padding =
            pad_image_to_target_size(&mut image, TEST_RESIZE_DIMENSIONS, TEST_PADDING_COLOR)
                .unwrap();

        // Assert
        assert_eq!(image.width(), TEST_RESIZE_DIMENSIONS.0);
        assert_eq!(image.height(), TEST_RESIZE_DIMENSIONS.1);
    }

    #[test]
    fn test_pad_image_pad_value() {
        // Arrange
        let image = synthetic_image();
        let mut image = resize_image(&image, TEST_RESIZE_DIMENSIONS);
        assert_eq!(image.width(), TEST_RESIZE_DIMENSIONS.0);

        // Act
        let _padding =
            pad_image_to_target_size(&mut image, TEST_RESIZE_DIMENSIONS, TEST_PADDING_COLOR)
                .unwrap();

        // Assert
        // left and right most pixels are of PAD_VALUE
        assert_eq!(image.get_pixel(0, 0).0[0], TEST_PADDING_COLOR[0]);
        assert_eq!(
            image.get_pixel(image.width() - 1, 0).0[0],
            TEST_PADDING_COLOR[0]
        );
        assert_eq!(
            image.get_pixel(0, image.height() - 1).0[0],
            TEST_PADDING_COLOR[0]
        );
        assert_eq!(
            image.get_pixel(image.width() - 1, image.height() - 1).0[0],
            TEST_PADDING_COLOR[0]
        );
    }

    #[test]
    #[should_panic(expected = "Cannot pad image with dimensions smaller than image dimensions")]
    fn test_pad_image_padding_dimensions_smaller_than_image() {
        // Arrange
        const PAD_DIMENSIONS: (u32, u32) = (
            TEST_RESIZE_DIMENSIONS.0 - 100,
            TEST_RESIZE_DIMENSIONS.1 - 100,
        );
        let image = synthetic_image();
        let mut image = resize_image(&image, TEST_RESIZE_DIMENSIONS);
        assert_eq!(image.width(), TEST_RESIZE_DIMENSIONS.0);

        // Act
        let _padding =
            pad_image_to_target_size(&mut image, PAD_DIMENSIONS, TEST_PADDING_COLOR).unwrap();
    }

    #[test]
    fn test_rgb_to_bgr() {
        // Arrange
        let image = synthetic_image();

        // Act
        let bgr_image = switch_br_channles(&image);

        // Assert
        for (pixel_rgb, pixel_bgr) in image.pixels().zip(bgr_image.pixels()) {
            let coord_rgb = pixel_rgb.2;
            let coords_bgr = pixel_bgr.2;
            assert_eq!(coord_rgb[0], coords_bgr[2]);
            assert_eq!(coord_rgb[1], coords_bgr[1]);
            assert_eq!(coord_rgb[2], coords_bgr[0]);
        }
    }

    #[test]
    fn test_image_to_tensor() {
        // Arrange
        let image = synthetic_image();
        let image = resize_image(&image, TEST_RESIZE_DIMENSIONS);
        let mut image = switch_br_channles(&image);
        let _padding =
            pad_image_to_target_size(&mut image, TEST_RESIZE_DIMENSIONS, TEST_PADDING_COLOR)
                .unwrap();
        let target_shape = (1, 640, 640, 3);
        let normalization_type = NormalizationType::Linear;

        // Act
        let tensor = raw_rgb_to_normalized_tensor(
            image.to_rgb8().as_raw(),
            image.width(),
            image.height(),
            &normalization_type,
            &target_shape,
        )
        .unwrap();

        // Assert
        assert_eq!(tensor.shape()[0], target_shape.0);
        assert_eq!(tensor.shape()[1], target_shape.1);
        assert_eq!(tensor.shape()[2], target_shape.2);
        assert_eq!(tensor.shape()[3], target_shape.3);

        // Check that values are normalized (should be roughly centered around 0)
        let mean_val = tensor.mean().unwrap();
        assert!(
            mean_val.abs() < 1.0,
            "Normalized values should be centered around 0"
        );
    }
}
