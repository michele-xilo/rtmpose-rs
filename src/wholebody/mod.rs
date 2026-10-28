pub mod config;
pub mod estimator;
pub mod visualization;

pub use estimator::RtmwPoseEstimator;

use crate::detector::RtmDetector;
use crate::tracker::GenericTracker;

pub type RtmwPoseTracker = GenericTracker<RtmDetector, RtmwPoseEstimator>;
