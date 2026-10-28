use super::config::{N_WHOLEBODY_KEYPOINTS, RTMW_L_MODEL_NAME, RTMW_M_MODEL_NAME};
use crate::config::{
    IMAGENET_MEAN_0_255, IMAGENET_STD_0_255, RTMPOSE_INPUT_DIMENSIONS, RTMPOSE_INPUT_TENSOR_NAME,
    RTMPOSE_INPUT_TENSOR_SHAPE, RTMPOSE_SIMCC_SPLIT_RATIO, RTMPOSE_SIMCC_X_OUTPUT,
    RTMPOSE_SIMCC_Y_OUTPUT,
};
use crate::execution::with_default_execution_providers;
use crate::postprocess::{apply_inverse_affine, decode_simcc};
use crate::shared::affine_warping::{get_affine_transform_matrix, warp_affine};
use crate::shared::image::raw_rgb_to_normalized_tensor;
use crate::shared::types::{
    BorderMode, EstimatorTrait, Landmark3D, MeanStdNormalizationConfig, ModelSize,
    NormalizationType, PoseResult, RegionOfInterest,
};
use anyhow::Result;
use image::DynamicImage;
#[cfg(not(target_os = "linux"))]
use ort::memory::Allocator;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;
use tracing::error;

// RTMW shares its input shape, preprocessing, and SimCC head layout with body
// RTMPose; only the model file (K=133 vs K=17) differs. We deliberately keep
// this estimator standalone — body RTMPose is the load-bearing path and
// shouldn't be refactored to share guts with wholebody.
pub struct RtmwPoseEstimator {
    session: Session,
    #[cfg(not(target_os = "linux"))]
    io_binding: ort::session::IoBinding,
    #[cfg(not(target_os = "linux"))]
    input_tensor: Tensor<f32>,
}

impl EstimatorTrait for RtmwPoseEstimator {
    fn new(model_size: ModelSize, models_dir: impl Into<PathBuf>) -> Result<Self> {
        let model_filename = Self::to_model_name(model_size);
        let model_path = models_dir.into().join(model_filename);

        let session = with_default_execution_providers(Session::builder()?)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e: ort::Error<_>| ort::Error::<()>::from(e))?
            .commit_from_file(&model_path)?;

        #[cfg(not(target_os = "linux"))]
        let (io_binding, input_tensor) = {
            let mut io_binding = session.create_binding()?;
            let allocator = Allocator::default();
            let s = RTMPOSE_INPUT_TENSOR_SHAPE;
            let input_shape = [s.0 as i64, s.1 as i64, s.2 as i64, s.3 as i64];
            let input_tensor = Tensor::<f32>::new(&allocator, input_shape)?;
            io_binding.bind_input(RTMPOSE_INPUT_TENSOR_NAME, &input_tensor)?;
            let memory_info = allocator.memory_info();
            io_binding.bind_output_to_device(RTMPOSE_SIMCC_X_OUTPUT, memory_info)?;
            io_binding.bind_output_to_device(RTMPOSE_SIMCC_Y_OUTPUT, memory_info)?;
            (io_binding, input_tensor)
        };

        Ok(Self {
            session,
            #[cfg(not(target_os = "linux"))]
            io_binding,
            #[cfg(not(target_os = "linux"))]
            input_tensor,
        })
    }

    fn to_model_name(model_size: ModelSize) -> String {
        // Medium/Full/Large/Heavy → RTMW-l (higher AP); everything else → M.
        // X variant intentionally not exposed: it ships at 384×288, which
        // would need a separate estimator path.
        match model_size {
            ModelSize::Medium | ModelSize::Full | ModelSize::Large | ModelSize::Heavy => {
                RTMW_L_MODEL_NAME.to_string()
            }
            _ => RTMW_M_MODEL_NAME.to_string(),
        }
    }

    fn infer(
        &mut self,
        frame: &DynamicImage,
        roi: &RegionOfInterest,
    ) -> Result<Option<PoseResult>> {
        let rgb = frame.to_rgb8();
        let (full_w, full_h) = rgb.dimensions();
        let raw = rgb.as_raw();

        let (transform_matrix, _rotation_matrix) =
            get_affine_transform_matrix(roi, RTMPOSE_INPUT_DIMENSIONS);
        let warped = warp_affine(
            raw,
            (full_w as usize, full_h as usize),
            &transform_matrix,
            (
                RTMPOSE_INPUT_DIMENSIONS.0 as usize,
                RTMPOSE_INPUT_DIMENSIONS.1 as usize,
            ),
            BorderMode::Constant(0),
        );
        let normalization = NormalizationType::MeanStd(MeanStdNormalizationConfig {
            mean: IMAGENET_MEAN_0_255,
            std: IMAGENET_STD_0_255,
        });
        let input_data = raw_rgb_to_normalized_tensor(
            &warped,
            RTMPOSE_INPUT_DIMENSIONS.0,
            RTMPOSE_INPUT_DIMENSIONS.1,
            &normalization,
            &RTMPOSE_INPUT_TENSOR_SHAPE,
        )?;

        #[cfg(not(target_os = "linux"))]
        let outputs = {
            let (_, tensor_data) = self.input_tensor.extract_tensor_mut();
            let contiguous;
            let slice = match input_data.as_slice() {
                Some(s) => s,
                None => {
                    contiguous = input_data.as_standard_layout().into_owned();
                    contiguous
                        .as_slice()
                        .ok_or_else(|| anyhow::anyhow!("rtmw input tensor not contiguous"))?
                }
            };
            tensor_data.copy_from_slice(slice);
            self.io_binding.synchronize_inputs()?;
            self.session.run_binding(&self.io_binding)?
        };

        #[cfg(target_os = "linux")]
        let outputs = {
            let input_tensor = Tensor::from_array(input_data)?;
            self.session
                .run(ort::inputs![RTMPOSE_INPUT_TENSOR_NAME => input_tensor])?
        };

        let simcc_x = outputs
            .get(RTMPOSE_SIMCC_X_OUTPUT)
            .ok_or_else(|| anyhow::anyhow!("RTMW output `{}` missing", RTMPOSE_SIMCC_X_OUTPUT))?
            .try_extract_array::<f32>()?;
        let simcc_y = outputs
            .get(RTMPOSE_SIMCC_Y_OUTPUT)
            .ok_or_else(|| anyhow::anyhow!("RTMW output `{}` missing", RTMPOSE_SIMCC_Y_OUTPUT))?
            .try_extract_array::<f32>()?;

        let x_shape = simcc_x.shape();
        let y_shape = simcc_y.shape();
        if x_shape.len() != 3 || y_shape.len() != 3 {
            error!(
                "RTMW SimCC outputs must be rank-3, got x={:?} y={:?}",
                x_shape, y_shape
            );
            return Ok(None);
        }
        if x_shape[1] != N_WHOLEBODY_KEYPOINTS || y_shape[1] != N_WHOLEBODY_KEYPOINTS {
            error!(
                "RTMW expected {} keypoints, got x={} y={}",
                N_WHOLEBODY_KEYPOINTS, x_shape[1], y_shape[1]
            );
            return Ok(None);
        }

        let x_view = simcc_x
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow::anyhow!("simcc_x dimensionality cast failed: {e}"))?;
        let y_view = simcc_y
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow::anyhow!("simcc_y dimensionality cast failed: {e}"))?;

        let decoded = decode_simcc(x_view, y_view, RTMPOSE_SIMCC_SPLIT_RATIO)?;

        let mut landmarks: Vec<Landmark3D> = Vec::with_capacity(decoded.len());
        let mut confidence_sum = 0.0_f32;
        for (warped_x, warped_y, score) in decoded {
            let (px, py) = match apply_inverse_affine(&transform_matrix, warped_x, warped_y) {
                Some(p) => p,
                None => {
                    error!("Failed to invert affine for RTMW keypoint");
                    return Ok(None);
                }
            };
            let confidence = 1.0 / (1.0 + (-score).exp());
            confidence_sum += confidence;
            landmarks.push(Landmark3D::new(px, py, 0.0, confidence, Some(confidence)));
        }
        let average_confidence = if landmarks.is_empty() {
            0.0
        } else {
            confidence_sum / landmarks.len() as f32
        };

        Ok(Some(PoseResult {
            pixel_landmarks: landmarks,
            average_confidence,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_model_name_lite_picks_m() {
        assert_eq!(
            RtmwPoseEstimator::to_model_name(ModelSize::Lite),
            RTMW_M_MODEL_NAME
        );
        assert_eq!(
            RtmwPoseEstimator::to_model_name(ModelSize::Nano),
            RTMW_M_MODEL_NAME
        );
    }

    #[test]
    fn test_to_model_name_medium_and_above_picks_l() {
        for size in [
            ModelSize::Medium,
            ModelSize::Full,
            ModelSize::Large,
            ModelSize::Heavy,
        ] {
            assert_eq!(RtmwPoseEstimator::to_model_name(size), RTMW_L_MODEL_NAME);
        }
    }

    #[test]
    #[ignore]
    fn test_rtmw_estimator_construction() {
        let est = RtmwPoseEstimator::new(ModelSize::Lite, "./models");
        assert!(
            est.is_ok(),
            "RtmwPoseEstimator construction must succeed with RTMW-m ONNX present"
        );
    }
}
