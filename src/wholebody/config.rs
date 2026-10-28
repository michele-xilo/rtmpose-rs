#![allow(dead_code)]

// RTMW (RTMPose Wholebody) — COCO-Wholebody layout (133 keypoints).
// Model filenames follow the upstream ONNX SDK naming with the `_simcc_cocktail14`
// suffix dropped for brevity. Update if your downloaded archive differs.
pub const RTMW_M_MODEL_NAME: &str = "rtmw_m_simcc_cocktail14_256x192.onnx";
pub const RTMW_L_MODEL_NAME: &str = "rtmw_l_simcc_cocktail14_256x192.onnx";

// Total wholebody keypoints: 17 body + 6 feet + 68 face + 42 hands = 133.
pub const N_WHOLEBODY_KEYPOINTS: usize = 133;

// COCO-Wholebody keypoint index ranges. Half-open `[start, end)` for `for i in
// range.clone()` ergonomics elsewhere.
pub const BODY_RANGE: std::ops::Range<usize> = 0..17;
pub const FEET_RANGE: std::ops::Range<usize> = 17..23;
pub const FACE_RANGE: std::ops::Range<usize> = 23..91; // 68 face landmarks
pub const LEFT_HAND_RANGE: std::ops::Range<usize> = 91..112; // 21 keypoints
pub const RIGHT_HAND_RANGE: std::ops::Range<usize> = 112..133; // 21 keypoints

// Foot edges, expressed in absolute indices (offset by FEET_RANGE.start).
// COCO-Wholebody foot order per side: big toe, small toe, heel.
// Per-side edges: ankle-bigToe, ankle-heel, bigToe-smallToe.
// Ankle indices come from the body block: 15 = left ankle, 16 = right ankle.
pub const FOOT_EDGES: [(usize, usize); 6] = [
    // Left foot
    (15, 17), // ankle → big toe
    (15, 19), // ankle → heel
    (17, 18), // big toe → small toe
    // Right foot
    (16, 20), // ankle → big toe
    (16, 22), // ankle → heel
    (20, 21), // big toe → small toe
];

// Hand skeleton (mediapipe topology — same 21-vertex layout RTMW emits):
// 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
// Stored as relative offsets so we can reuse them for both hands by adding the
// base index (LEFT_HAND_RANGE.start or RIGHT_HAND_RANGE.start).
pub const HAND_EDGES_RELATIVE: [(usize, usize); 20] = [
    // Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    // Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    // Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    // Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    // Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

// Per-keypoint confidence threshold. Face/hand peaks tend to be weaker than
// body peaks on small subjects, so we use a slightly looser threshold than the
// body-only pipeline. Revisit after benchmarking on real data.
pub const RTMW_LANDMARK_CONFIDENCE_THRESHOLD: f32 = 0.25;
