use crate::select::select_camera;
use crate::theme;
use anyhow::{Context, Result};
use eframe::egui;
use image::{DynamicImage, RgbImage};
use nokhwa::utils::{CameraFormat, FrameFormat, Resolution};
use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    threaded::CallbackCamera,
    utils::{RequestedFormat, RequestedFormatType},
};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

const CAMERA_FRAME_RATE: u32 = 30;
const CAMERA_RESOLUTION: (u32, u32) = (1920, 1080);

pub fn run<F>(title: &str, heading: &str, mut annotate: F) -> Result<()>
where
    F: FnMut(&DynamicImage) -> Result<RgbImage> + Send + 'static,
{
    let image_data: Arc<Mutex<Option<egui::ColorImage>>> = Arc::new(Mutex::new(None));
    let image_data_clone = Arc::clone(&image_data);

    nokhwa_initialize(|granted| info!("Camera permission: {}", granted));
    let format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
            Resolution::new(CAMERA_RESOLUTION.0, CAMERA_RESOLUTION.1),
            FrameFormat::YUYV,
            CAMERA_FRAME_RATE,
        )));

    let selected = select_camera(&format)?;
    let mut camera = CallbackCamera::new(selected.index().clone(), format, move |buffer| {
        let resolution = buffer.resolution();
        let (w, h) = (resolution.width(), resolution.height());
        let rgb_buffer = match buffer.decode_image::<RgbFormat>() {
            Ok(b) => b,
            Err(e) => {
                error!("Failed to decode frame: {e}");
                return;
            }
        };
        let Some(rgb_image) = RgbImage::from_raw(w, h, rgb_buffer.as_raw().to_vec()) else {
            error!("Failed to build RgbImage from camera buffer");
            return;
        };
        let mut frame = DynamicImage::ImageRgb8(rgb_image);

        let start = std::time::Instant::now();
        match annotate(&frame) {
            Ok(overlay) => {
                info!("Inference + overlay time {:?}", start.elapsed());
                frame = DynamicImage::ImageRgb8(overlay);
            }
            Err(e) => error!("Pose tracking failed: {e:#}"),
        }

        let Some(color_image) = crate::frame::dynamic_image_to_color_image(&frame) else {
            error!("Camera frame size mismatch, dropping frame");
            return;
        };
        if let Ok(mut data) = image_data_clone.lock() {
            *data = Some(color_image);
        }
    })?;
    camera
        .open_stream()
        .context("Failed to open camera stream")?;

    let camera_resolution = camera
        .camera_format()
        .map(|f| f.resolution())
        .context("Failed to get camera format")?;
    let camera_width = camera_resolution.width() as f32;
    let camera_height = camera_resolution.height() as f32;
    info!("Camera streaming started with resolution {camera_width}x{camera_height}");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([camera_width, camera_height])
            .with_resizable(true)
            .with_title(title),
        ..Default::default()
    };

    let app = CameraApp {
        heading: heading.to_owned(),
        image_data,
        texture_handle: None,
    };
    eframe::run_native(
        title,
        native_options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe::run_native failed: {e}"))?;

    info!("Stopping camera stream");
    Ok(())
}

struct CameraApp {
    heading: String,
    image_data: Arc<Mutex<Option<egui::ColorImage>>>,
    texture_handle: Option<egui::TextureHandle>,
}

impl eframe::App for CameraApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(mut data) = self.image_data.lock()
            && let Some(color_image) = data.take()
        {
            self.texture_handle = Some(ctx.load_texture(
                "camera_frame",
                color_image,
                egui::TextureOptions::default(),
            ));
        }

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::BACKGROUND)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| {
                theme::panel_header(ui, "LIVE CAMERA", &self.heading, true);
                ui.add_space(12.0);

                theme::card_frame().show(ui, |ui| {
                    if let Some(texture) = &self.texture_handle {
                        let size = fit_within(texture.size_vec2(), ui.available_size());
                        ui.add(
                            egui::Image::from_texture(texture)
                                .fit_to_exact_size(size)
                                .corner_radius(egui::CornerRadius::same(12)),
                        );
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label(theme::body("Waiting for camera frame…"));
                        });
                    }
                });
            });

        ctx.request_repaint();
    }
}

fn fit_within(image: egui::Vec2, available: egui::Vec2) -> egui::Vec2 {
    let scale = (available.x / image.x).min(available.y / image.y);
    image * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_within_limited_by_height_keeps_aspect() {
        // Arrange
        let image = egui::vec2(1920.0, 1080.0);
        let available = egui::vec2(1920.0, 540.0);

        // Act
        let size = fit_within(image, available);

        // Assert
        assert_eq!(size, egui::vec2(960.0, 540.0));
    }

    #[test]
    fn test_fit_within_limited_by_width_keeps_aspect() {
        // Arrange
        let image = egui::vec2(1920.0, 1080.0);
        let available = egui::vec2(960.0, 1080.0);

        // Act
        let size = fit_within(image, available);

        // Assert
        assert_eq!(size, egui::vec2(960.0, 540.0));
    }
}
