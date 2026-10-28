use anyhow::{Context, Result, anyhow};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::query;
use nokhwa::utils::{ApiBackend, CameraFormat, CameraInfo, RequestedFormat, RequestedFormatType};
use tracing::{info, warn};

pub fn select_camera(requested: &RequestedFormat<'_>) -> Result<CameraInfo> {
    let cameras = query(ApiBackend::Auto).context("Failed to query cameras")?;
    if cameras.is_empty() {
        return Err(anyhow!("No cameras found"));
    }
    info!("Found {} camera(s)", cameras.len());

    let probes = probe_cameras(cameras);
    let picked = pick_camera(&probes, requested)
        .ok_or_else(|| anyhow!("No camera supports the requested format {requested:?}"))?;
    info!("Using camera: {}", picked.human_name());
    Ok(picked.clone())
}

// V4L2 exposes UVC metadata nodes (/dev/video1 on many webcams) as queryable
// cameras that yield no video formats, so the first entry is not always usable.
fn probe_cameras(cameras: Vec<CameraInfo>) -> Vec<(CameraInfo, Vec<CameraFormat>)> {
    cameras
        .into_iter()
        .map(|info| {
            let formats = compatible_formats(&info).unwrap_or_else(|e| {
                warn!("Skipping camera {}: {e}", info.human_name());
                Vec::new()
            });
            (info, formats)
        })
        .collect()
}

fn compatible_formats(info: &CameraInfo) -> Result<Vec<CameraFormat>> {
    let probe = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
    let mut camera = nokhwa::Camera::new(info.index().clone(), probe)?;
    Ok(camera.compatible_camera_formats()?)
}

pub fn pick_camera<'a>(
    probes: &'a [(CameraInfo, Vec<CameraFormat>)],
    requested: &RequestedFormat<'_>,
) -> Option<&'a CameraInfo> {
    probes
        .iter()
        .find(|(_, formats)| requested.fulfill(formats).is_some())
        .map(|(info, _)| info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, FrameFormat, RequestedFormatType, Resolution};

    fn info(index: u32) -> CameraInfo {
        CameraInfo::new("cam", "desc", "misc", CameraIndex::Index(index))
    }

    fn hd_yuyv() -> CameraFormat {
        CameraFormat::new(Resolution::new(1920, 1080), FrameFormat::YUYV, 30)
    }

    fn requested() -> RequestedFormat<'static> {
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(hd_yuyv()))
    }

    #[test]
    fn skips_camera_without_any_compatible_format() {
        // Arrange
        let probes = vec![(info(1), vec![]), (info(0), vec![hd_yuyv()])];

        // Act
        let picked = pick_camera(&probes, &requested());

        // Assert
        assert_eq!(picked.map(CameraInfo::index), Some(&CameraIndex::Index(0)));
    }

    #[test]
    fn skips_camera_whose_formats_cannot_fulfill_the_request() {
        // Arrange
        let mjpeg_only = CameraFormat::new(Resolution::new(1920, 1080), FrameFormat::MJPEG, 30);
        let probes = vec![(info(1), vec![mjpeg_only]), (info(0), vec![hd_yuyv()])];

        // Act
        let picked = pick_camera(&probes, &requested());

        // Assert
        assert_eq!(picked.map(CameraInfo::index), Some(&CameraIndex::Index(0)));
    }

    #[test]
    fn prefers_the_first_camera_that_can_fulfill() {
        // Arrange
        let probes = vec![(info(0), vec![hd_yuyv()]), (info(2), vec![hd_yuyv()])];

        // Act
        let picked = pick_camera(&probes, &requested());

        // Assert
        assert_eq!(picked.map(CameraInfo::index), Some(&CameraIndex::Index(0)));
    }

    #[test]
    fn returns_none_when_no_camera_can_fulfill() {
        // Arrange
        let probes = vec![(info(0), vec![]), (info(1), vec![])];

        // Act
        let picked = pick_camera(&probes, &requested());

        // Assert
        assert!(picked.is_none());
    }
}
