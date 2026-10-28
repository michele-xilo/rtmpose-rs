use crate::shared::colors::WHITE;
use crate::shared::types::Bbox;
use image::{Rgb as ImageRgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::rect::Rect;

pub const BBOX_COLOR: [u8; 3] = WHITE;
pub const BBOX_THICKNESS: i32 = 2;

// Draw a hollow rectangle with a given pixel thickness, clamping each expanded
// border to the image bounds. Shared by the RTMPose and RTMW renderers so both
// draw detection/ROI boxes identically.
pub fn draw_thick_bbox(
    img: &mut RgbImage,
    bbox: &Bbox,
    color: [u8; 3],
    thickness: i32,
    image_width: u32,
    image_height: u32,
) {
    let xyxy = bbox.to_xyxy();
    let x1 = xyxy.x1.max(0.0) as u32;
    let y1 = xyxy.y1.max(0.0) as u32;
    let x2 = xyxy.x2.min(image_width as f32 - 1.0).max(0.0) as u32;
    let y2 = xyxy.y2.min(image_height as f32 - 1.0).max(0.0) as u32;

    if x2 <= x1 || y2 <= y1 {
        return;
    }

    let rect = Rect::at(x1 as i32, y1 as i32).of_size(x2 - x1, y2 - y1);
    let color = ImageRgb(color);
    let frame = Rect::at(0, 0).of_size(image_width, image_height);
    for t in 0..thickness {
        if let Some(expanded) = Rect::at(rect.left() - t, rect.top() - t)
            .of_size(rect.width() + 2 * t as u32, rect.height() + 2 * t as u32)
            .intersect(frame)
        {
            draw_hollow_rect_mut(img, expanded, color);
        }
    }
}

pub const SKELETON_DOT_RADIUS_RATIO: f32 = 0.004;

pub fn skeleton_dot_radius(width: u32, height: u32) -> i32 {
    let diag = (width as f32).hypot(height as f32);
    (diag * SKELETON_DOT_RADIUS_RATIO).max(1.0).round() as i32
}

pub fn draw_skeleton_line(img: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 3]) {
    draw_line_segment_mut(img, (x0, y0), (x1, y1), ImageRgb(color));
}

pub fn draw_landmark_dot(img: &mut RgbImage, cx: f32, cy: f32, radius: i32, color: [u8; 3]) {
    draw_filled_circle_mut(
        img,
        (cx.round() as i32, cy.round() as i32),
        radius,
        ImageRgb(color),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::types::{Bbox, BboxXYXY};

    fn make_bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> Bbox {
        Bbox::BboxXYXY(BboxXYXY { x1, y1, x2, y2 })
    }

    #[test]
    fn test_skeleton_dot_radius_scales_with_diagonal() {
        // Arrange — 1280×720 diagonal ≈ 1469 px
        // Act
        let r = skeleton_dot_radius(1280, 720);
        // Assert — 1469 * 0.004 ≈ 5.88 → rounds to 6
        assert_eq!(r, 6);
    }

    #[test]
    fn test_skeleton_dot_radius_minimum_is_one() {
        // Arrange — tiny image where ratio would give < 1 px
        // Act
        let r = skeleton_dot_radius(1, 1);
        // Assert
        assert_eq!(r, 1);
    }

    #[test]
    fn test_draw_skeleton_line_colors_pixels_along_line() {
        // Arrange
        let mut img = RgbImage::from_pixel(30, 30, ImageRgb([0, 0, 0]));
        let color = [255, 0, 0];

        // Act — horizontal line across the middle
        draw_skeleton_line(&mut img, 5.0, 15.0, 25.0, 15.0, color);

        // Assert — pixels on the line are colored
        assert_eq!(img.get_pixel(10, 15).0, color);
        assert_eq!(img.get_pixel(20, 15).0, color);
    }

    #[test]
    fn test_draw_skeleton_line_does_not_color_off_line_pixels() {
        // Arrange
        let mut img = RgbImage::from_pixel(30, 30, ImageRgb([0, 0, 0]));

        // Act — horizontal line at y=15
        draw_skeleton_line(&mut img, 5.0, 15.0, 25.0, 15.0, [255, 0, 0]);

        // Assert — row above and below are untouched
        assert_eq!(img.get_pixel(15, 14).0, [0, 0, 0]);
        assert_eq!(img.get_pixel(15, 16).0, [0, 0, 0]);
    }

    #[test]
    fn test_draw_landmark_dot_colors_center_pixel() {
        // Arrange
        let mut img = RgbImage::from_pixel(20, 20, ImageRgb([0, 0, 0]));
        let color = [0, 255, 0];

        // Act
        draw_landmark_dot(&mut img, 10.0, 10.0, 2, color);

        // Assert — center must be colored regardless of radius
        assert_eq!(img.get_pixel(10, 10).0, color);
    }

    #[test]
    fn test_draw_landmark_dot_fills_within_radius() {
        // Arrange
        let mut img = RgbImage::from_pixel(30, 30, ImageRgb([0, 0, 0]));
        let color = [0, 0, 255];

        // Act
        draw_landmark_dot(&mut img, 15.0, 15.0, 3, color);

        // Assert — pixels within radius are colored, far pixels are not
        assert_eq!(img.get_pixel(15, 15).0, color);
        assert_eq!(img.get_pixel(15, 17).0, color); // distance 2, inside radius 3
        assert_eq!(img.get_pixel(15, 19).0, [0, 0, 0]); // distance 4, outside radius 3
    }

    #[test]
    fn test_draw_thick_bbox_writes_pixels_inside_bounds() {
        // Arrange
        let mut img = RgbImage::from_pixel(50, 40, ImageRgb([0, 0, 0]));
        let bbox = make_bbox(10.0, 10.0, 30.0, 20.0);

        // Act
        draw_thick_bbox(&mut img, &bbox, [255, 0, 0], 2, 50, 40);

        // Assert — at least one red pixel landed on the rectangle border.
        let any_red = img.pixels().any(|p| p.0 == [255, 0, 0]);
        assert!(any_red, "Expected at least one red border pixel");
    }

    #[test]
    fn test_draw_thick_bbox_noop_for_empty_rect() {
        // Arrange — degenerate bbox (x2 == x1) should be a no-op.
        let mut img = RgbImage::from_pixel(20, 20, ImageRgb([0, 0, 0]));
        let bbox = make_bbox(5.0, 5.0, 5.0, 10.0);

        // Act
        draw_thick_bbox(&mut img, &bbox, [255, 0, 0], 2, 20, 20);

        // Assert — image untouched.
        assert!(img.pixels().all(|p| p.0 == [0, 0, 0]));
    }

    #[test]
    fn test_draw_thick_bbox_clamps_to_image_bounds() {
        // Arrange — bbox extends past the image; must not panic.
        let mut img = RgbImage::from_pixel(20, 20, ImageRgb([0, 0, 0]));
        let bbox = make_bbox(-5.0, -5.0, 25.0, 25.0);

        // Act
        draw_thick_bbox(&mut img, &bbox, [0, 255, 0], 3, 20, 20);

        // Assert — some green pixel on the visible portion of the border.
        let any_green = img.pixels().any(|p| p.0 == [0, 255, 0]);
        assert!(any_green);
    }
}
