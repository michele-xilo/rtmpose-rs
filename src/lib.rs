pub mod config;
pub mod detector;
pub mod estimator;
pub mod execution;
pub mod postprocess;
pub mod shared;
pub mod tracker;
pub mod visualization;
pub mod wholebody;

pub use detector::RtmDetector;
pub use estimator::RtmPoseEstimator;
pub use tracker::{GenericTracker, PoseTrackerOptions, TrackerTrait};
pub use wholebody::{RtmwPoseEstimator, RtmwPoseTracker};

pub type RtmPoseTracker = GenericTracker<RtmDetector, RtmPoseEstimator>;
