use crate::shared::types::{
    DetectorTrait, EstimatorTrait, ModelSize, PoseResult, RegionOfInterest,
};
use anyhow::Result;
use image::DynamicImage;
use std::path::PathBuf;
use std::time::Instant;
use tracing::warn;

#[derive(Clone, Debug)]
pub struct PoseTrackerOptions {
    pub model_size: ModelSize,
    pub models_dir: PathBuf,
    pub max_n_poses: usize,
}

impl PoseTrackerOptions {
    pub fn new(model_size: ModelSize, models_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_size,
            models_dir: models_dir.into(),
            max_n_poses: DEFAULT_MAX_POSES,
        }
    }

    pub fn with_max_n_poses(mut self, max_n_poses: usize) -> Self {
        self.max_n_poses = max_n_poses;
        self
    }
}

const MAX_INTERVAL_FOR_SKIPPING_DETECTION_MS: u128 = 500;

const SUBCAPACITY_DETECTOR_INTERVAL_MS: u128 = 500;

const POSE_OVERLAP_IOU_THRESHOLD: f32 = 0.5;

pub const DEFAULT_MAX_POSES: usize = 1;

pub trait TrackerTrait {
    fn track(&mut self, image: &DynamicImage) -> Result<Vec<Option<PoseResult>>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PoseId(pub u64);

struct TrackedPose {
    id: PoseId,
    roi: RegionOfInterest,
}

fn new_tracked_pose(id: PoseId, roi: RegionOfInterest) -> TrackedPose {
    TrackedPose { id, roi }
}

pub struct GenericTracker<D, E> {
    detector: D,
    estimator: E,
    tracked_poses: Vec<TrackedPose>,
    t_previous_frame: Option<Instant>,
    last_detection_at: Option<Instant>,
    max_n_poses: usize,
    next_pose_id: u64,
    // Positional PoseIds corresponding to the last `track()` return value.
    last_emitted_ids: Vec<PoseId>,
}

impl<D, E> GenericTracker<D, E>
where
    D: DetectorTrait,
    E: EstimatorTrait,
{
    pub fn new(options: PoseTrackerOptions) -> Result<Self> {
        let max_n_poses = options.max_n_poses.max(1);
        Ok(Self {
            detector: D::new(options.models_dir.clone())?,
            estimator: E::new(options.model_size, options.models_dir)?,
            tracked_poses: Vec::new(),
            t_previous_frame: None,
            last_detection_at: None,
            max_n_poses,
            next_pose_id: 0,
            last_emitted_ids: Vec::new(),
        })
    }

    pub fn last_emitted_ids(&self) -> &[PoseId] {
        &self.last_emitted_ids
    }

    pub fn current_rois(&self) -> impl Iterator<Item = (PoseId, &RegionOfInterest)> + '_ {
        self.tracked_poses.iter().map(|t| (t.id, &t.roi))
    }

    pub fn current_roi(&self, pose_id: PoseId) -> Option<&RegionOfInterest> {
        self.tracked_poses
            .iter()
            .find(|t| t.id == pose_id)
            .map(|t| &t.roi)
    }

    fn mint_id(&mut self) -> PoseId {
        let id = PoseId(self.next_pose_id);
        self.next_pose_id = self.next_pose_id.wrapping_add(1);
        id
    }
}

impl<D, E> TrackerTrait for GenericTracker<D, E>
where
    D: DetectorTrait,
    E: EstimatorTrait,
{
    fn track(&mut self, frame: &DynamicImage) -> Result<Vec<Option<PoseResult>>> {
        let now = Instant::now();
        let last_roi_was_too_long_ago = self
            .t_previous_frame
            .map(|t| (now - t).as_millis() > MAX_INTERVAL_FOR_SKIPPING_DETECTION_MS)
            .unwrap_or(false);
        self.t_previous_frame = Some(now);

        if last_roi_was_too_long_ago {
            // Tracking gap: drop everything and re-seed with fresh ids.
            warn!("Frame gap exceeded — re-running the pose detector");
            self.tracked_poses.clear();
            self.run_detector_and_seed(frame, now)?;
        } else if self.tracked_poses.is_empty() {
            // Nothing to reuse: look for poses immediately (no throttle).
            self.run_detector_and_seed(frame, now)?;
        } else if self.tracked_poses.len() < self.max_n_poses && self.should_top_up_detect(now) {
            // Below capacity: merge in any newly-detected, non-overlapping poses.
            self.run_detector_and_merge(frame, now)?;
        }

        if self.tracked_poses.is_empty() {
            self.last_emitted_ids.clear();
            return Ok(vec![]);
        }

        // Snapshot ids in positional order BEFORE inference — the returned
        // vector is aligned with this slice whether or not individual poses
        // estimate successfully.
        self.last_emitted_ids = self.tracked_poses.iter().map(|t| t.id).collect();

        let rois: Vec<RegionOfInterest> =
            self.tracked_poses.iter().map(|t| t.roi.clone()).collect();
        let mut pose_results: Vec<Option<PoseResult>> = Vec::with_capacity(rois.len());
        for roi in &rois {
            pose_results.push(self.estimator.infer(frame, roi)?);
        }

        let frame_width = frame.width();
        let frame_height = frame.height();

        let old_tracked = std::mem::take(&mut self.tracked_poses);

        if pose_results.is_empty() {
            return Ok(pose_results);
        }

        let mut output: Vec<Option<PoseResult>> = Vec::with_capacity(pose_results.len());
        let mut next_tracked: Vec<TrackedPose> = Vec::with_capacity(pose_results.len());

        // Partial retention: a single failed pose drops only its own ROI —
        // siblings that estimated successfully keep tracking (and keep their
        // PoseId) instead of forcing the detector to re-seed everyone.
        for (pose_opt, tracked) in pose_results.into_iter().zip(old_tracked) {
            let Some(pose) = pose_opt else {
                output.push(None);
                continue;
            };

            let next_roi = RegionOfInterest::from_pose_result(&pose, frame_width, frame_height)
                .map_err(|e| warn!("Failed to create ROI from pose result: {}", e))
                .ok();

            if let Some(next_roi) = next_roi {
                next_tracked.push(TrackedPose {
                    id: tracked.id,
                    roi: next_roi,
                });
            }

            output.push(Some(pose));
        }

        self.tracked_poses = next_tracked;
        self.dedup_collapsed_poses();
        Ok(output)
    }
}

impl<D, E> GenericTracker<D, E>
where
    D: DetectorTrait,
    E: EstimatorTrait,
{
    fn dedup_collapsed_poses(&mut self) {
        if self.max_n_poses <= 1 {
            return;
        }
        let mut survivors: Vec<TrackedPose> = Vec::with_capacity(self.tracked_poses.len());
        for pose in self.tracked_poses.drain(..) {
            let overlaps_survivor = survivors
                .iter()
                .any(|s| s.roi.bbox.calculate_iou(&pose.roi.bbox) > POSE_OVERLAP_IOU_THRESHOLD);
            if !overlaps_survivor {
                survivors.push(pose);
            }
        }
        self.tracked_poses = survivors;
    }

    fn should_top_up_detect(&self, now: Instant) -> bool {
        self.last_detection_at
            .map(|t| (now - t).as_millis() >= SUBCAPACITY_DETECTOR_INTERVAL_MS)
            .unwrap_or(true)
    }

    fn run_detector_and_seed(&mut self, frame: &DynamicImage, now: Instant) -> Result<()> {
        let detections = self.detector.infer(frame)?;
        self.last_detection_at = Some(now);
        for roi in detections.regions_of_interest.into_iter() {
            if self.tracked_poses.len() >= self.max_n_poses {
                break;
            }
            let id = self.mint_id();
            self.tracked_poses.push(new_tracked_pose(id, roi));
        }
        Ok(())
    }

    fn run_detector_and_merge(&mut self, frame: &DynamicImage, now: Instant) -> Result<()> {
        let detections = self.detector.infer(frame)?;
        self.last_detection_at = Some(now);
        for roi in detections.regions_of_interest.into_iter() {
            if self.tracked_poses.len() >= self.max_n_poses {
                break;
            }
            let overlaps_existing = self
                .tracked_poses
                .iter()
                .any(|t| t.roi.bbox.calculate_iou(&roi.bbox) > POSE_OVERLAP_IOU_THRESHOLD);
            if !overlaps_existing {
                let id = self.mint_id();
                self.tracked_poses.push(new_tracked_pose(id, roi));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::types::{Bbox, BboxXYWH, DetectorOutput, Landmark3D, ModelSize, PoseResult};
    use image::{DynamicImage, RgbImage};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct MockDetector {
        call_count: Arc<Mutex<usize>>,
        return_rois: Vec<RegionOfInterest>,
    }

    impl MockDetector {
        fn new_with_rois(rois: Vec<RegionOfInterest>) -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                return_rois: rois,
            }
        }

        fn get_call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    impl DetectorTrait for MockDetector {
        fn new(_model_dir: impl Into<PathBuf>) -> Result<Self> {
            Ok(Self {
                call_count: Arc::new(Mutex::new(0)),
                return_rois: vec![create_test_roi()],
            })
        }

        fn infer(&mut self, _frame: &DynamicImage) -> Result<DetectorOutput> {
            *self.call_count.lock().unwrap() += 1;
            Ok(DetectorOutput {
                regions_of_interest: self.return_rois.clone(),
            })
        }
    }

    #[derive(Clone)]
    struct MockEstimator {
        call_count: Arc<Mutex<usize>>,
        return_poses: Vec<Option<PoseResult>>,
    }

    impl MockEstimator {
        fn new_with_poses(poses: Vec<Option<PoseResult>>) -> Self {
            Self {
                call_count: Arc::new(Mutex::new(0)),
                return_poses: poses,
            }
        }

        fn get_call_count(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    impl EstimatorTrait for MockEstimator {
        fn new(_model_size: ModelSize, _model_dir: impl Into<PathBuf>) -> Result<Self> {
            Ok(Self {
                call_count: Arc::new(Mutex::new(0)),
                return_poses: vec![Some(create_test_pose_result())],
            })
        }

        fn to_model_name(_model_size: ModelSize) -> String {
            String::from("mock_model_name")
        }

        fn infer(
            &mut self,
            _frame: &DynamicImage,
            _roi: &RegionOfInterest,
        ) -> Result<Option<PoseResult>> {
            let mut count = self.call_count.lock().unwrap();
            let idx = *count;
            *count += 1;
            // Cycle through the seeded result(s). If no poses were seeded,
            // emit None for every call.
            if self.return_poses.is_empty() {
                return Ok(None);
            }
            Ok(self.return_poses[idx % self.return_poses.len()].clone())
        }
    }

    fn create_test_image() -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::new(640, 480))
    }

    fn create_test_roi() -> RegionOfInterest {
        RegionOfInterest {
            bbox: Bbox::BboxXYWH(BboxXYWH {
                center_x: 100.0,
                center_y: 100.0,
                width: 50.0,
                height: 50.0,
            }),
            confidence: 0.9,
            landmarks: None,
            rotation: None,
        }
    }

    fn roi_at(cx: f32, cy: f32) -> RegionOfInterest {
        RegionOfInterest {
            bbox: Bbox::BboxXYWH(BboxXYWH {
                center_x: cx,
                center_y: cy,
                width: 50.0,
                height: 50.0,
            }),
            confidence: 0.9,
            landmarks: None,
            rotation: None,
        }
    }

    fn create_test_pose_result() -> PoseResult {
        create_test_pose_result_at(100.0, 100.0)
    }

    fn create_test_pose_result_at(cx: f32, cy: f32) -> PoseResult {
        PoseResult {
            pixel_landmarks: vec![
                Landmark3D::new(cx, cy, 0.0, 0.9, None),
                Landmark3D::new(cx + 50.0, cy + 50.0, 0.0, 0.8, None),
            ],
            average_confidence: 0.85,
        }
    }

    fn tracker_with_mocks<D, E>(
        detector: D,
        estimator: E,
        tracked_poses: Vec<TrackedPose>,
        t_previous_frame: Option<Instant>,
        max_n_poses: usize,
    ) -> GenericTracker<D, E> {
        GenericTracker {
            detector,
            estimator,
            tracked_poses,
            t_previous_frame,
            last_detection_at: t_previous_frame,
            max_n_poses,
            next_pose_id: 100,
            last_emitted_ids: Vec::new(),
        }
    }

    #[test]
    fn test_track_calls_detector_when_no_tracked_poses() {
        // Arrange
        let detector = MockDetector::new("test_model_dir").unwrap();
        let estimator = MockEstimator::new(ModelSize::Lite, "test_model_dir").unwrap();
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let result = tracker.track(&image);

        // Assert
        assert!(result.is_ok());
        assert_eq!(
            detector.get_call_count(),
            1,
            "Detector should be called when no tracked poses exist"
        );
        assert_eq!(estimator.get_call_count(), 1, "Estimator should be called");
    }

    #[test]
    fn test_current_rois_empty_before_any_tracking() {
        // Arrange
        let detector = MockDetector::new_with_rois(vec![]);
        let estimator = MockEstimator::new(ModelSize::Lite, "test_model_dir").unwrap();
        let tracker = tracker_with_mocks(detector, estimator, Vec::new(), None, 1);

        // Act
        let rois: Vec<_> = tracker.current_rois().collect();

        // Assert
        assert!(rois.is_empty());
    }

    #[test]
    fn test_current_rois_returns_next_frame_rois_after_track() {
        // Arrange — a fresh tracker; detector seeds one ROI, estimator returns a
        // pose so the tracker keeps a TrackedPose for the next frame.
        let detector = MockDetector::new_with_rois(vec![create_test_roi()]);
        let estimator = MockEstimator::new_with_poses(vec![Some(create_test_pose_result())]);
        let mut tracker = tracker_with_mocks(detector, estimator, Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();
        let rois: Vec<_> = tracker.current_rois().collect();

        // Assert — one tracked pose remains; its ROI is the expanded bbox
        // derived from the pose's landmarks (so it differs from the seed ROI).
        assert_eq!(rois.len(), 1);
        let (id, roi) = rois[0];
        assert_eq!(tracker.current_roi(id).map(|r| r.bbox), Some(roi.bbox));
    }

    #[test]
    fn test_current_rois_empty_when_pose_estimation_fails() {
        // Arrange — detector seeds an ROI but the estimator returns None,
        // so the tracker drops the pose and has nothing to expose.
        let detector = MockDetector::new_with_rois(vec![create_test_roi()]);
        let estimator = MockEstimator::new_with_poses(vec![None]);
        let mut tracker = tracker_with_mocks(detector, estimator, Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();
        let rois: Vec<_> = tracker.current_rois().collect();

        // Assert
        assert!(rois.is_empty());
    }

    #[test]
    fn test_track_skips_detector_when_tracked_poses_exist() {
        // Arrange
        let detector = MockDetector::new_with_rois(vec![]);
        let estimator = MockEstimator::new(ModelSize::Lite, "test_model_dir").unwrap();
        let tracked = vec![new_tracked_pose(PoseId(1), create_test_roi())];
        let mut tracker = tracker_with_mocks(
            detector.clone(),
            estimator.clone(),
            tracked,
            Some(Instant::now()),
            1,
        );
        let image = create_test_image();

        // Act
        let result = tracker.track(&image);

        // Assert
        assert!(result.is_ok());
        assert_eq!(
            detector.get_call_count(),
            0,
            "Detector should be skipped when at capacity and timing gap is small"
        );
        assert_eq!(
            estimator.get_call_count(),
            1,
            "Estimator should still be called"
        );
    }

    #[test]
    fn test_track_reruns_detector_when_gap_exceeds_threshold() {
        // Arrange — tracked poses exist but the last frame is older than
        // MAX_INTERVAL_FOR_SKIPPING_DETECTION_MS, so the detector must re-run.
        let detector = MockDetector::new_with_rois(vec![create_test_roi()]);
        let estimator = MockEstimator::new_with_poses(vec![Some(create_test_pose_result())]);
        let tracked = vec![new_tracked_pose(PoseId(1), create_test_roi())];
        let stale = Instant::now()
            - std::time::Duration::from_millis(
                (MAX_INTERVAL_FOR_SKIPPING_DETECTION_MS as u64) + 100,
            );
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), tracked, Some(stale), 1);
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(
            detector.get_call_count(),
            1,
            "Detector should re-run when the frame gap exceeds the skip threshold"
        );
    }

    #[test]
    fn test_track_resets_tracked_poses_when_pose_is_lost() {
        // Arrange
        let detector = MockDetector::new("test_model_dir").unwrap();
        let estimator = MockEstimator::new_with_poses(vec![None]);
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let result1 = tracker.track(&image);
        assert!(result1.is_ok());
        let poses = result1.unwrap();
        assert_eq!(poses.len(), 1, "Should return one pose slot");
        assert!(poses[0].is_none(), "Pose should be None (lost)");

        let result2 = tracker.track(&image);
        assert!(result2.is_ok());

        // Assert
        assert_eq!(
            detector.get_call_count(),
            2,
            "Detector should be called twice — once initially and once after tracked poses cleared due to lost pose"
        );
    }

    #[test]
    fn test_track_handles_empty_regions_of_interest() {
        // Arrange
        let detector = MockDetector::new_with_rois(vec![]);
        let estimator = MockEstimator::new(ModelSize::Lite, "test_model_dir").unwrap();
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let result = tracker.track(&image).unwrap();

        // Assert
        assert!(result.is_empty(), "Should return empty vector when no ROIs");
        assert_eq!(detector.get_call_count(), 1, "Detector should be called");
        assert_eq!(
            estimator.get_call_count(),
            0,
            "Estimator should not be called for empty ROIs"
        );
    }

    #[test]
    fn test_track_derives_next_roi_from_raw_landmarks() {
        // Arrange — pose result with aux landmarks so next-frame ROI is derivable.
        let detector = MockDetector::new_with_rois(vec![create_test_roi()]);
        let estimator = MockEstimator::new_with_poses(vec![Some(create_test_pose_result())]);
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();

        // Assert — after a successful track, tracked_poses should be populated
        // with a fresh ROI derived from the pose result.
        assert_eq!(
            tracker.tracked_poses.len(),
            1,
            "Tracker should retain the pose for the next frame"
        );
    }

    // ───────── multi-person (max_n_poses > 1) ─────────

    #[test]
    fn test_detector_cap_respected_at_initial_seed() {
        // Arrange — detector returns 3 candidates, tracker cap is 2. Each slot
        // must return landmarks at a distinct position so dedup doesn't fire.
        let detector = MockDetector::new_with_rois(vec![
            roi_at(100.0, 100.0),
            roi_at(300.0, 300.0),
            roi_at(500.0, 500.0),
        ]);
        let estimator = MockEstimator::new_with_poses(vec![
            Some(create_test_pose_result_at(100.0, 100.0)),
            Some(create_test_pose_result_at(300.0, 300.0)),
        ]);
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), Vec::new(), None, 2);
        let image = create_test_image();

        // Act
        let result = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(result.len(), 2, "Only max_n_poses slots should be emitted");
        assert_eq!(
            tracker.tracked_poses.len(),
            2,
            "Tracker must retain exactly max_n_poses poses"
        );
    }

    #[test]
    fn test_subcapacity_detector_throttle_skips_within_interval() {
        // Arrange — tracker at 1/3, last detection was recent. Top-up should be
        // throttled and the detector must NOT run.
        let detector = MockDetector::new_with_rois(vec![roi_at(500.0, 500.0)]);
        let estimator = MockEstimator::new_with_poses(vec![Some(create_test_pose_result())]);
        let tracked = vec![new_tracked_pose(PoseId(1), create_test_roi())];
        let mut tracker = tracker_with_mocks(
            detector.clone(),
            estimator.clone(),
            tracked,
            Some(Instant::now()),
            3,
        );
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(
            detector.get_call_count(),
            0,
            "Detector must be throttled when last top-up was recent"
        );
    }

    #[test]
    fn test_subcapacity_detector_runs_after_interval_and_merges() {
        // Arrange — one tracked pose, detector returns a non-overlapping second
        // candidate, and the last detection was longer than the throttle ago.
        // Each slot returns landmarks at its own position so dedup doesn't fire.
        let detector = MockDetector::new_with_rois(vec![roi_at(500.0, 500.0)]);
        let estimator = MockEstimator::new_with_poses(vec![
            Some(create_test_pose_result_at(100.0, 100.0)),
            Some(create_test_pose_result_at(500.0, 500.0)),
        ]);
        let tracked = vec![new_tracked_pose(PoseId(1), create_test_roi())];
        let stale = Instant::now()
            - std::time::Duration::from_millis((SUBCAPACITY_DETECTOR_INTERVAL_MS as u64) + 50);
        let mut tracker = GenericTracker {
            detector: detector.clone(),
            estimator: estimator.clone(),
            tracked_poses: tracked,
            t_previous_frame: Some(Instant::now()),
            last_detection_at: Some(stale),
            max_n_poses: 2,
            next_pose_id: 100,
            last_emitted_ids: Vec::new(),
        };
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();

        // Assert
        assert_eq!(
            detector.get_call_count(),
            1,
            "Detector should run after the throttle interval elapses"
        );
        assert_eq!(
            tracker.tracked_poses.len(),
            2,
            "Non-overlapping detection should be merged"
        );
    }

    #[test]
    fn test_subcapacity_detector_skips_overlapping_detection() {
        // Arrange — one tracked pose at (100,100); detector returns an
        // overlapping detection at the same centre. Merge must drop it.
        let detector = MockDetector::new_with_rois(vec![roi_at(100.0, 100.0)]);
        let estimator = MockEstimator::new_with_poses(vec![Some(create_test_pose_result())]);
        let tracked = vec![new_tracked_pose(PoseId(1), create_test_roi())];
        let stale = Instant::now()
            - std::time::Duration::from_millis((SUBCAPACITY_DETECTOR_INTERVAL_MS as u64) + 50);
        let mut tracker = GenericTracker {
            detector: detector.clone(),
            estimator: estimator.clone(),
            tracked_poses: tracked,
            t_previous_frame: Some(Instant::now()),
            last_detection_at: Some(stale),
            max_n_poses: 2,
            next_pose_id: 100,
            last_emitted_ids: Vec::new(),
        };
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();

        // Assert — still a single tracked pose after the merge attempt.
        assert_eq!(
            tracker.tracked_poses.len(),
            1,
            "Overlapping top-up detection must be rejected"
        );
    }

    #[test]
    fn test_pose_id_stable_across_successful_frames() {
        // Arrange
        let detector = MockDetector::new_with_rois(vec![create_test_roi()]);
        let estimator = MockEstimator::new_with_poses(vec![Some(create_test_pose_result())]);
        let mut tracker =
            tracker_with_mocks(detector.clone(), estimator.clone(), Vec::new(), None, 1);
        let image = create_test_image();

        // Act
        let _ = tracker.track(&image).unwrap();
        let id_frame_1 = tracker.tracked_poses[0].id;
        let _ = tracker.track(&image).unwrap();
        let id_frame_2 = tracker.tracked_poses[0].id;

        // Assert
        assert_eq!(
            id_frame_1, id_frame_2,
            "PoseId must persist across frames when the pose is continuously tracked"
        );
    }

    #[test]
    fn test_pose_id_fresh_after_loss_and_reacquire() {
        // Arrange — first frame finds a pose, second frame loses it, third
        // frame finds one again. The third-frame id must differ from the first.
        struct ToggleEstimator {
            call_count: Arc<Mutex<usize>>,
        }
        impl EstimatorTrait for ToggleEstimator {
            fn new(_: ModelSize, _: impl Into<PathBuf>) -> Result<Self> {
                unreachable!()
            }
            fn to_model_name(_: ModelSize) -> String {
                String::new()
            }
            fn infer(
                &mut self,
                _: &DynamicImage,
                _: &RegionOfInterest,
            ) -> Result<Option<PoseResult>> {
                let mut n = self.call_count.lock().unwrap();
                *n += 1;
                // Second call loses the pose; others succeed.
                if *n == 2 {
                    Ok(None)
                } else {
                    Ok(Some(create_test_pose_result()))
                }
            }
        }

        let detector = MockDetector::new_with_rois(vec![create_test_roi()]);
        let estimator = ToggleEstimator {
            call_count: Arc::new(Mutex::new(0)),
        };
        let mut tracker = tracker_with_mocks(detector, estimator, Vec::new(), None, 1);
        let image = create_test_image();

        let _ = tracker.track(&image).unwrap();
        let first_id = tracker.tracked_poses[0].id;
        // Frame 2: pose lost, tracked_poses cleared.
        let _ = tracker.track(&image).unwrap();
        assert!(tracker.tracked_poses.is_empty());
        // Frame 3: detector re-seeds, fresh id.
        let _ = tracker.track(&image).unwrap();
        let reacquired_id = tracker.tracked_poses[0].id;

        assert_ne!(
            first_id, reacquired_id,
            "A re-acquired pose should get a fresh PoseId, not the old one"
        );
    }

    #[test]
    fn test_last_emitted_ids_align_with_returned_results() {
        // Arrange — detector returns 3 ROIs, cap is 3. Estimator's second ROI
        // returns None; first and third succeed. `last_emitted_ids` must have
        // length 3 and line up positionally with the returned slots (including
        // the None slot), so that a downstream consumer can zip them.
        let detector = MockDetector::new_with_rois(vec![
            roi_at(100.0, 100.0),
            roi_at(300.0, 300.0),
            roi_at(500.0, 500.0),
        ]);
        struct SequenceEstimator {
            call_count: Arc<Mutex<usize>>,
        }
        impl EstimatorTrait for SequenceEstimator {
            fn new(_: ModelSize, _: impl Into<PathBuf>) -> Result<Self> {
                unreachable!()
            }
            fn to_model_name(_: ModelSize) -> String {
                String::new()
            }
            fn infer(
                &mut self,
                _: &DynamicImage,
                _: &RegionOfInterest,
            ) -> Result<Option<PoseResult>> {
                let mut n = self.call_count.lock().unwrap();
                let idx = *n;
                *n += 1;
                Ok(match idx {
                    0 => Some(create_test_pose_result()),
                    1 => None,
                    _ => Some(create_test_pose_result()),
                })
            }
        }
        let estimator = SequenceEstimator {
            call_count: Arc::new(Mutex::new(0)),
        };
        let mut tracker = tracker_with_mocks(detector, estimator, Vec::new(), None, 3);
        let image = create_test_image();

        // Act
        let results = tracker.track(&image).unwrap();
        let ids: Vec<PoseId> = tracker.last_emitted_ids().to_vec();

        // Assert
        assert_eq!(results.len(), 3);
        assert_eq!(ids.len(), 3, "last_emitted_ids must align with results");
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_some());
        // All three ids must be distinct (freshly minted this frame).
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[0], ids[2]);
    }

    #[test]
    fn test_collapsed_tracked_poses_are_deduplicated() {
        // Arrange — two slots whose ROIs sit on top of each other (IoU >> threshold),
        // simulating both estimators converging on the same person. The tracker is
        // at capacity so the top-up path never fires, but after dedup it should
        // drop back to one slot and then allow the top-up detect to search for a
        // second person.
        let detector = MockDetector::new_with_rois(vec![roi_at(500.0, 500.0)]);
        let estimator = MockEstimator::new_with_poses(vec![
            Some(create_test_pose_result()),
            Some(create_test_pose_result()),
        ]);
        let overlapping_roi = roi_at(100.0, 100.0);
        let tracked = vec![
            new_tracked_pose(PoseId(1), overlapping_roi.clone()),
            new_tracked_pose(PoseId(2), overlapping_roi.clone()),
        ];
        let mut tracker = GenericTracker {
            detector: detector.clone(),
            estimator: estimator.clone(),
            tracked_poses: tracked,
            t_previous_frame: Some(Instant::now()),
            last_detection_at: Some(
                Instant::now()
                    - std::time::Duration::from_millis(
                        (SUBCAPACITY_DETECTOR_INTERVAL_MS as u64) + 50,
                    ),
            ),
            max_n_poses: 2,
            next_pose_id: 100,
            last_emitted_ids: Vec::new(),
        };
        let image = create_test_image();

        // Act — after this frame the duplicate slot must be evicted, leaving
        // the tracker below capacity so the top-up detector can fire next frame.
        let _ = tracker.track(&image).unwrap();

        // Assert — only one slot survives; the other was a duplicate.
        assert!(
            tracker.tracked_poses.len() < 2,
            "Collapsed tracked poses must be deduplicated; got {} slots",
            tracker.tracked_poses.len()
        );
    }

    #[test]
    fn test_last_emitted_ids_cleared_when_no_detections() {
        // Arrange — detector finds nothing, so nothing is emitted.
        let detector = MockDetector::new_with_rois(vec![]);
        let estimator = MockEstimator::new(ModelSize::Lite, "test_model_dir").unwrap();
        let mut tracker = tracker_with_mocks(detector, estimator, Vec::new(), None, 2);
        let image = create_test_image();

        // Act
        let results = tracker.track(&image).unwrap();

        // Assert
        assert!(results.is_empty());
        assert!(
            tracker.last_emitted_ids().is_empty(),
            "Empty return should leave last_emitted_ids empty"
        );
    }

    // ───────── PoseTrackerOptions ─────────

    #[test]
    fn test_options_new_sets_defaults() {
        // Arrange / Act
        let opts = PoseTrackerOptions::new(ModelSize::Lite, "./models");

        // Assert
        assert_eq!(opts.max_n_poses, DEFAULT_MAX_POSES);
    }

    #[test]
    fn test_options_with_max_n_poses_overrides_default() {
        // Arrange / Act
        let opts = PoseTrackerOptions::new(ModelSize::Lite, "./models").with_max_n_poses(3);

        // Assert
        assert_eq!(opts.max_n_poses, 3);
    }
}
