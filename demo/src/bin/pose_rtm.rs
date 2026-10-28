use anyhow::Result;
use rtmpose::shared::types::ModelSize;
use rtmpose::tracker::TrackerTrait;
use rtmpose::visualization::draw_rtm_predictions_on_image;
use rtmpose::{PoseTrackerOptions, RtmPoseTracker};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let options = PoseTrackerOptions::new(ModelSize::Medium, "./models").with_max_n_poses(1);
    let mut tracker = RtmPoseTracker::new(options)?;

    demo::app::run("RTMPose", "RTMPose (COCO-17)", move |frame| {
        let pose_results = tracker.track(frame)?;
        draw_rtm_predictions_on_image(frame, &pose_results, None, Some(8.0))
    })
}
