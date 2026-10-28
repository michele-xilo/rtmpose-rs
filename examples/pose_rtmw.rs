use anyhow::{anyhow, Result};
use rtmpose::shared::image::load_image_from_path;
use rtmpose::shared::types::ModelSize;
use rtmpose::tracker::TrackerTrait;
use rtmpose::wholebody::visualization::draw_rtmw_predictions_on_image;
use rtmpose::{PoseTrackerOptions, RtmwPoseTracker};
use tracing::info;

const POSE_OUTPUT_PATH: &str = "assets/outputs";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let image_input_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: pose_rtmw <image>"))?;
    // Lite -> RTMW-m (fastest), Medium/Heavy -> RTMW-l (higher AP).
    let model_size = ModelSize::Lite;
    let max_n_poses = 2;

    let options =
        PoseTrackerOptions::new(model_size.clone(), "./models").with_max_n_poses(max_n_poses);
    let mut tracker = RtmwPoseTracker::new(options)?;
    info!("Initialised RtmwPoseTracker (model_size={model_size}, max_n_poses={max_n_poses})");

    let frame = load_image_from_path(&image_input_path)?;
    let pose_results = tracker.track(&frame)?;
    info!(
        "Successfully inferred RTMW wholebody for {} person(s)",
        pose_results.iter().filter(|p| p.is_some()).count()
    );

    std::fs::create_dir_all(POSE_OUTPUT_PATH)?;
    let out = draw_rtmw_predictions_on_image(&frame, &pose_results, None, None)?;
    let stem = std::path::Path::new(&image_input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    out.save(format!("{POSE_OUTPUT_PATH}/pose_rtmw_{stem}.png"))?;

    Ok(())
}
