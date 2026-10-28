// Canonical COCO-17 skeleton topology. Lives in `shared` so any layer
// (pose trackers, action_recognition) references one source of truth instead of
// reaching into `pose::rtm` internals.

// COCO-17 keypoint ordering. Index → name.
pub const COCO_KEYPOINT_NAMES: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];
pub const N_COCO_KEYPOINTS: usize = COCO_KEYPOINT_NAMES.len();

// COCO-17 skeleton edges, used by the visualizer to draw the stick figure.
pub const COCO_SKELETON_EDGES: [(usize, usize); 19] = [
    // Face
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    // Shoulders / arms
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    // Torso
    (5, 11),
    (6, 12),
    (11, 12),
    // Legs
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    // Cross links (shoulder-hip diagonals + ear-shoulder), draw last so they
    // sit underneath
    (3, 5),
    (4, 6),
    (0, 5),
];

// Named COCO-17 joints so callers index by meaning, not magic numbers. The
// discriminants are the canonical COCO indices, so `kp as usize` indexes a
// keypoint array directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum CocoKeypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl CocoKeypoint {
    pub const ALL: [CocoKeypoint; N_COCO_KEYPOINTS] = [
        CocoKeypoint::Nose,
        CocoKeypoint::LeftEye,
        CocoKeypoint::RightEye,
        CocoKeypoint::LeftEar,
        CocoKeypoint::RightEar,
        CocoKeypoint::LeftShoulder,
        CocoKeypoint::RightShoulder,
        CocoKeypoint::LeftElbow,
        CocoKeypoint::RightElbow,
        CocoKeypoint::LeftWrist,
        CocoKeypoint::RightWrist,
        CocoKeypoint::LeftHip,
        CocoKeypoint::RightHip,
        CocoKeypoint::LeftKnee,
        CocoKeypoint::RightKnee,
        CocoKeypoint::LeftAnkle,
        CocoKeypoint::RightAnkle,
    ];

    pub fn name(self) -> &'static str {
        COCO_KEYPOINT_NAMES[self as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_discriminants_match_name_table() {
        // Arrange / Act / Assert
        for kp in CocoKeypoint::ALL {
            assert_eq!(kp.name(), COCO_KEYPOINT_NAMES[kp as usize]);
        }
    }

    #[test]
    fn all_covers_every_index_in_order() {
        // Arrange / Act / Assert
        for (i, kp) in CocoKeypoint::ALL.iter().enumerate() {
            assert_eq!(*kp as usize, i);
        }
    }
}
