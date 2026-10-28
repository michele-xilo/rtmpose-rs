use anyhow::Result;
use rtmpose::shared::types::ModelSize;
use rtmpose::tracker::TrackerTrait;
use rtmpose::wholebody::visualization::draw_rtmw_predictions_on_image;
use rtmpose::{PoseTrackerOptions, RtmwPoseTracker};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Lite -> RTMW-m (fastest), Medium/Large/Heavy -> RTMW-l (higher AP).
    let options = PoseTrackerOptions::new(ModelSize::Lite, "./models").with_max_n_poses(1);
    let mut tracker = RtmwPoseTracker::new(options)?;

    demo::app::run(
        "RTMW Wholebody",
        "RTMW Wholebody (COCO-133)",
        move |frame| {
            let pose_results = tracker.track(frame)?;
            draw_rtmw_predictions_on_image(frame, &pose_results, None, Some(6.0))
        },
    )
}
