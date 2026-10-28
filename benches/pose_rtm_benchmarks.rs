use criterion::{black_box, criterion_group, criterion_main, Criterion};

use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array3;
use rand::Rng;
use rtmpose::config::{
    COCO_SKELETON_EDGES, IMAGENET_MEAN_0_255, IMAGENET_STD_0_255, N_COCO_KEYPOINTS,
    RTMPOSE_INPUT_DIMENSIONS, RTMPOSE_INPUT_TENSOR_SHAPE, RTMPOSE_SIMCC_SPLIT_RATIO,
};
use rtmpose::postprocess::decode_simcc;
use rtmpose::shared::affine_warping::{get_affine_transform_matrix, warp_affine};
use rtmpose::shared::image::raw_rgb_to_normalized_tensor;
use rtmpose::shared::types::{
    Bbox, BboxXYWH, BorderMode, Landmark3D, MeanStdNormalizationConfig, NormalizationType,
    PoseResult, RegionOfInterest,
};
use rtmpose::visualization::draw_rtm_predictions_on_image;
use rtmpose::wholebody::config::N_WHOLEBODY_KEYPOINTS;
use rtmpose::wholebody::visualization::draw_rtmw_predictions_on_image;

fn make_hd_frame() -> DynamicImage {
    let mut rng = rand::rng();
    let buf: Vec<u8> = (0..1920 * 1080 * 3).map(|_| rng.random::<u8>()).collect();
    DynamicImage::ImageRgb8(RgbImage::from_raw(1920, 1080, buf).unwrap())
}

fn make_roi() -> RegionOfInterest {
    RegionOfInterest {
        bbox: Bbox::BboxXYWH(BboxXYWH {
            center_x: 960.0,
            center_y: 540.0,
            width: 600.0,
            height: 900.0,
        }),
        confidence: 0.95,
        landmarks: None,
        rotation: None,
    }
}

fn bench_warp_affine(c: &mut Criterion) {
    // Arrange
    let frame = make_hd_frame();
    let rgb = frame.to_rgb8();
    let raw = rgb.as_raw();
    let roi = make_roi();
    let (transform, _) = get_affine_transform_matrix(&roi, RTMPOSE_INPUT_DIMENSIONS);

    // Act
    c.bench_function("rtmpose_warp_affine_hd_to_192x256", |b| {
        b.iter(|| {
            black_box(warp_affine(
                black_box(raw),
                (1920, 1080),
                black_box(&transform),
                (
                    RTMPOSE_INPUT_DIMENSIONS.0 as usize,
                    RTMPOSE_INPUT_DIMENSIONS.1 as usize,
                ),
                BorderMode::Constant(0),
            ))
        })
    });
}

fn bench_normalize_tensor(c: &mut Criterion) {
    // Arrange
    let mut rng = rand::rng();
    let raw: Vec<u8> = (0..RTMPOSE_INPUT_DIMENSIONS.0 * RTMPOSE_INPUT_DIMENSIONS.1 * 3)
        .map(|_| rng.random::<u8>())
        .collect();
    let normalization = NormalizationType::MeanStd(MeanStdNormalizationConfig {
        mean: IMAGENET_MEAN_0_255,
        std: IMAGENET_STD_0_255,
    });

    // Act
    c.bench_function("rtmpose_normalize_imagenet_192x256", |b| {
        b.iter(|| {
            black_box(
                raw_rgb_to_normalized_tensor(
                    black_box(&raw),
                    RTMPOSE_INPUT_DIMENSIONS.0,
                    RTMPOSE_INPUT_DIMENSIONS.1,
                    &normalization,
                    &RTMPOSE_INPUT_TENSOR_SHAPE,
                )
                .unwrap(),
            )
        })
    });
}

fn bench_decode_simcc(c: &mut Criterion) {
    // Arrange — fully populated SimCC logits with sharp peaks.
    let w = (RTMPOSE_INPUT_DIMENSIONS.0 as f32 * RTMPOSE_SIMCC_SPLIT_RATIO) as usize;
    let h = (RTMPOSE_INPUT_DIMENSIONS.1 as f32 * RTMPOSE_SIMCC_SPLIT_RATIO) as usize;
    let mut rng = rand::rng();
    let x = Array3::from_shape_fn((1, N_COCO_KEYPOINTS, w), |_| rng.random::<f32>());
    let y = Array3::from_shape_fn((1, N_COCO_KEYPOINTS, h), |_| rng.random::<f32>());

    // Act
    c.bench_function("rtmpose_decode_simcc_17kpt", |b| {
        b.iter(|| black_box(decode_simcc(x.view(), y.view(), RTMPOSE_SIMCC_SPLIT_RATIO)))
    });
}

fn bench_draw_rtm(c: &mut Criterion) {
    // Arrange
    let frame = DynamicImage::ImageRgb8(RgbImage::from_pixel(1280, 720, Rgb([128, 128, 128])));
    let mut rng = rand::rng();
    let lms: Vec<Landmark3D> = (0..N_COCO_KEYPOINTS)
        .map(|_| {
            Landmark3D::new(
                rng.random::<f32>() * 1280.0,
                rng.random::<f32>() * 720.0,
                0.0,
                0.9,
                Some(0.9),
            )
        })
        .collect();
    let pose = PoseResult {
        pixel_landmarks: lms,
        average_confidence: 0.9,
    };
    let poses = vec![Some(pose)];
    // Touch the skeleton constant so we get a compile error if it's renamed.
    let _edges = COCO_SKELETON_EDGES.len();

    // Act
    c.bench_function("rtmpose_draw_predictions_720p", |b| {
        b.iter(|| black_box(draw_rtm_predictions_on_image(&frame, &poses, None, None).unwrap()))
    });
}

fn bench_decode_simcc_wholebody(c: &mut Criterion) {
    // Arrange — same generator as body, but K=133. The decoder is K-agnostic;
    // this bench measures the per-keypoint scaling of the postprocess on
    // wholebody SimCC outputs.
    let w = (RTMPOSE_INPUT_DIMENSIONS.0 as f32 * RTMPOSE_SIMCC_SPLIT_RATIO) as usize;
    let h = (RTMPOSE_INPUT_DIMENSIONS.1 as f32 * RTMPOSE_SIMCC_SPLIT_RATIO) as usize;
    let mut rng = rand::rng();
    let x = Array3::from_shape_fn((1, N_WHOLEBODY_KEYPOINTS, w), |_| rng.random::<f32>());
    let y = Array3::from_shape_fn((1, N_WHOLEBODY_KEYPOINTS, h), |_| rng.random::<f32>());

    // Act
    c.bench_function("rtmw_decode_simcc_133kpt", |b| {
        b.iter(|| black_box(decode_simcc(x.view(), y.view(), RTMPOSE_SIMCC_SPLIT_RATIO)))
    });
}

fn bench_draw_rtmw(c: &mut Criterion) {
    // Arrange — 133 keypoints, all above confidence threshold so every body /
    // foot / hand edge fires and every face dot draws. Worst case for the
    // visualiser since face dots are 1 px each.
    let frame = DynamicImage::ImageRgb8(RgbImage::from_pixel(1280, 720, Rgb([128, 128, 128])));
    let mut rng = rand::rng();
    let lms: Vec<Landmark3D> = (0..N_WHOLEBODY_KEYPOINTS)
        .map(|_| {
            Landmark3D::new(
                rng.random::<f32>() * 1280.0,
                rng.random::<f32>() * 720.0,
                0.0,
                0.9,
                Some(0.9),
            )
        })
        .collect();
    let pose = PoseResult {
        pixel_landmarks: lms,
        average_confidence: 0.9,
    };
    let poses = vec![Some(pose)];

    // Act
    c.bench_function("rtmw_draw_predictions_720p", |b| {
        b.iter(|| black_box(draw_rtmw_predictions_on_image(&frame, &poses, None, None).unwrap()))
    });
}

criterion_group!(
    benches,
    bench_warp_affine,
    bench_normalize_tensor,
    bench_decode_simcc,
    bench_draw_rtm,
    bench_decode_simcc_wholebody,
    bench_draw_rtmw,
);
criterion_main!(benches);
