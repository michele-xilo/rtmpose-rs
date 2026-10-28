use egui::{Color32, ColorImage, Vec2};
use image::DynamicImage;

pub fn raw_rgb_to_color_image(raw: &[u8], width: usize, height: usize) -> Option<ColorImage> {
    let expected = width * height * 3;
    if raw.len() < expected {
        return None;
    }
    let mut pixels = Vec::with_capacity(width * height);
    for chunk in raw[..expected].as_chunks::<3>().0 {
        pixels.push(Color32::from_rgb(chunk[0], chunk[1], chunk[2]));
    }
    Some(ColorImage {
        source_size: Vec2::new(width as f32, height as f32),
        size: [width, height],
        pixels,
    })
}

pub fn dynamic_image_to_color_image(img: &DynamicImage) -> Option<ColorImage> {
    let rgb = img.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    raw_rgb_to_color_image(rgb.as_raw(), width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_rgb_to_color_image_strips_stride_padding() {
        // Arrange
        let width: usize = 4;
        let height: usize = 3;
        // Camera HAL adds one extra row of stride padding (same as MX Brio 705 at 1920x1080)
        let padded_len = (width * height + width) * 3;
        let raw = vec![128u8; padded_len];

        // Act
        let result = raw_rgb_to_color_image(&raw, width, height);

        // Assert
        assert!(result.is_some());
        assert_eq!(result.unwrap().pixels.len(), width * height);
    }

    #[test]
    fn test_raw_rgb_to_color_image_exact_size() {
        // Arrange
        let width: usize = 2;
        let height: usize = 2;
        let raw = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

        // Act
        let result = raw_rgb_to_color_image(&raw, width, height);

        // Assert
        assert!(result.is_some());
        let img = result.unwrap();
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels[0], Color32::from_rgb(10, 20, 30));
        assert_eq!(img.pixels[3], Color32::from_rgb(100, 110, 120));
    }

    #[test]
    fn test_raw_rgb_to_color_image_returns_none_when_too_small() {
        // Arrange
        let raw = vec![0u8; 5];

        // Act
        let result = raw_rgb_to_color_image(&raw, 2, 2);

        // Assert
        assert!(result.is_none());
    }
}
