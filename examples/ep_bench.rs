// Diagnostic benchmark for RTMPose and RTMDet under different ORT execution
// provider configurations and IO binding options.
//
// Usage:
//   cargo run --release --example ep_bench -- [model] [ep] [io]
//
// Arguments:
//   model  estimator-m | estimator-w-m | estimator-w-l | detector  (default: estimator-m)
//   ep     default | cpu
//          | coreml-nn-cpuandgpu | coreml-nn-cpuonly | coreml-nn-all
//          | coreml-mlp-cpuandgpu | coreml-mlp-cpuonly | coreml-mlp-all
//          | cuda | cuda-graph | cuda-tf32-nhwc
//                                                               (default: default)
//   io     no-binding | io-binding                             (default: no-binding)
//
// "default" is the current production config (CoreML NN + CPUAndGPU on macOS aarch64).
// coreml-nn-*   NeuralNetwork format; -cpuandgpu/-cpuonly/-all controls compute units.
// coreml-mlp-*  MLProgram format; same compute-unit suffix variants.
// io-binding    pre-allocates the input tensor and uses IoBinding (mirrors production
//               RTMPose estimator path); no-binding creates a fresh Tensor each run.
// cuda-graph    Linux only, and currently rejected: capture needs a stable bound input
//               address, i.e. io-binding, which is unsound on CUDA (see below). It also
//               fails session creation outright on models whose graph is not fully
//               CUDA-partitioned (rtmdet), and there is no graph + nhwc variant:
//               prefer_nhwc inserts CPU-side layout-transform nodes, which un-partitions
//               the graph and breaks capture.
// On CUDA, io-binding is downgraded to no-binding: the bound tensor lives in host memory
// and is only uploaded once, so binding measures the model running on stale input.
use anyhow::{anyhow, Result};
use ndarray::Array4;
use ort::memory::Allocator;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use rtmpose::config::{
    RTMDET_INPUT_TENSOR_NAME, RTMDET_INPUT_TENSOR_SHAPE, RTMDET_NANO_MODEL_NAME,
    RTMDET_OUTPUT_DETS, RTMDET_OUTPUT_LABELS, RTMPOSE_INPUT_TENSOR_NAME,
    RTMPOSE_INPUT_TENSOR_SHAPE, RTMPOSE_M_MODEL_NAME, RTMPOSE_SIMCC_X_OUTPUT,
    RTMPOSE_SIMCC_Y_OUTPUT,
};
use rtmpose::execution::with_default_execution_providers;
use rtmpose::wholebody::config::{RTMW_L_MODEL_NAME, RTMW_M_MODEL_NAME};
use std::time::{Duration, Instant};

const N_WARMUP: usize = 5;
const N_BENCH: usize = 50;

#[cfg(target_os = "macos")]
fn build_coreml_session(
    model_path: &str,
    cpu_only: bool,
    use_all: bool,
    mlprogram: bool,
) -> Result<Session> {
    use ort::execution_providers::coreml::{ComputeUnits, ModelFormat};
    use ort::execution_providers::CoreMLExecutionProvider;

    let compute_units = if cpu_only {
        ComputeUnits::CPUOnly
    } else if use_all {
        ComputeUnits::All
    } else {
        ComputeUnits::CPUAndGPU
    };

    let provider = {
        let p = CoreMLExecutionProvider::default()
            .with_compute_units(compute_units)
            .with_static_input_shapes(true);
        if mlprogram {
            p.with_model_format(ModelFormat::MLProgram)
        } else {
            p
        }
    };

    Session::builder()
        .map_err(|e| anyhow!("{e:?}"))?
        .with_execution_providers([provider.build()])
        .map_err(|e| anyhow!("{e:?}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e:?}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("{e:?}"))
}

// Graph capture requires every node on the CUDA EP and a stable bound input
// address, so it only ever pairs with `io-binding`. Registering with
// `error_on_failure` keeps a missing cuDNN 9 from masquerading as a slow GPU.
//
// Options are only ever set when enabling them. ONNX Runtime enables `use_tf32`
// by default (ort's docs claim otherwise), so passing `with_tf32(false)`
// actively turns TF32 tensor cores off and costs 3.3x on matmul-heavy models.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn build_cuda_session(model_path: &str, graph: bool, tf32: bool, nhwc: bool) -> Result<Session> {
    use ort::ep::CUDA;

    let mut provider = CUDA::default();
    if graph {
        provider = provider.with_cuda_graph(true);
    }
    if tf32 {
        provider = provider.with_tf32(true);
    }
    if nhwc {
        provider = provider.with_prefer_nhwc(true);
    }

    Session::builder()
        .map_err(|e| anyhow!("{e:?}"))?
        .with_execution_providers([provider.build().error_on_failure()])
        .map_err(|e| anyhow!("{e:?}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e:?}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("{e:?}"))
}

// `Allocator::default()` is a HOST allocator, so binding its tensor to a CUDA session
// copies host->device once at bind time; later `extract_tensor_mut` writes never reach
// the GPU and every iteration re-runs the bind-time contents. Measured proof: two
// different inputs give bit-identical outputs on CUDA, non-zero diff on CPU. Reporting
// that path as a timing would understate cost by a whole per-frame upload, so CUDA falls
// back to `Session::run`. A correct binding needs a device-resident input plus an
// explicit per-frame H2D memcpy, which ort exposes no API for.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn resolve_io_mode(ep: &str, use_io_binding: bool) -> Result<bool> {
    if !ep.starts_with("cuda") {
        return Ok(use_io_binding);
    }
    if ep == "cuda-graph" {
        return Err(anyhow!(
            "ep 'cuda-graph' is unavailable: replay needs a stable bound input address, and CUDA \
             io-binding feeds the model stale bind-time data. Re-enable once the input tensor is \
             device-resident and updated per frame."
        ));
    }
    // Printed rather than logged: `fmt::init()` defaults to ERROR, so a `warn!` here
    // would be invisible in the common invocation.
    if use_io_binding {
        println!(
            "  WARNING: io-binding is stale on CUDA (no per-frame host->device copy); benchmarking with Session::run instead"
        );
    }
    Ok(false)
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn resolve_io_mode(_ep: &str, use_io_binding: bool) -> Result<bool> {
    Ok(use_io_binding)
}

fn build_session(model_path: &str, ep: &str) -> Result<Session> {
    let builder = Session::builder().map_err(|e| anyhow!("{e:?}"))?;
    let builder = match ep {
        "default" => with_default_execution_providers(builder).map_err(|e| anyhow!("{e:?}"))?,
        "cpu" => builder,
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        "cuda" => return build_cuda_session(model_path, false, false, false),
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        "cuda-graph" => return build_cuda_session(model_path, true, false, false),
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        "cuda-tf32-nhwc" => return build_cuda_session(model_path, false, true, true),
        #[cfg(target_os = "macos")]
        "coreml-nn-cpuandgpu" => return build_coreml_session(model_path, false, false, false),
        #[cfg(target_os = "macos")]
        "coreml-nn-cpuonly" => return build_coreml_session(model_path, true, false, false),
        #[cfg(target_os = "macos")]
        "coreml-nn-all" => return build_coreml_session(model_path, false, true, false),
        #[cfg(target_os = "macos")]
        "coreml-mlp-cpuandgpu" => return build_coreml_session(model_path, false, false, true),
        #[cfg(target_os = "macos")]
        "coreml-mlp-cpuonly" => return build_coreml_session(model_path, true, false, true),
        #[cfg(target_os = "macos")]
        "coreml-mlp-all" => return build_coreml_session(model_path, false, true, true),
        other => return Err(anyhow!("unknown ep '{other}'")),
    };
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e:?}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("{e:?}"))
}

fn bench_run(
    session: &mut Session,
    input_name: &str,
    shape: (usize, usize, usize, usize),
) -> Result<Duration> {
    let input: Array4<f32> = Array4::zeros((shape.0, shape.1, shape.2, shape.3));
    for _ in 0..N_WARMUP {
        let t = Tensor::from_array(input.clone())?;
        let _ = session.run(ort::inputs![input_name => t])?;
    }
    let start = Instant::now();
    for _ in 0..N_BENCH {
        let t = Tensor::from_array(input.clone())?;
        let _ = session.run(ort::inputs![input_name => t])?;
    }
    Ok(start.elapsed() / N_BENCH as u32)
}

fn bench_io_binding(
    session: &mut Session,
    input_name: &str,
    output_names: &[&str],
    shape: (usize, usize, usize, usize),
) -> Result<Duration> {
    let allocator = Allocator::default();
    let input_shape = [
        shape.0 as i64,
        shape.1 as i64,
        shape.2 as i64,
        shape.3 as i64,
    ];
    let mut input_tensor = Tensor::<f32>::new(&allocator, input_shape)?;

    let mut io_binding = session.create_binding()?;
    io_binding.bind_input(input_name, &input_tensor)?;
    let mem_info = allocator.memory_info();
    for &name in output_names {
        io_binding.bind_output_to_device(name, mem_info)?;
    }

    let n_elems = shape.0 * shape.1 * shape.2 * shape.3;
    let zeros = vec![0.0f32; n_elems];

    for _ in 0..N_WARMUP {
        {
            let (_, data) = input_tensor.extract_tensor_mut();
            data.copy_from_slice(&zeros);
        }
        io_binding.synchronize_inputs()?;
        session.run_binding(&io_binding)?;
    }
    let start = Instant::now();
    for _ in 0..N_BENCH {
        {
            let (_, data) = input_tensor.extract_tensor_mut();
            data.copy_from_slice(&zeros);
        }
        io_binding.synchronize_inputs()?;
        session.run_binding(&io_binding)?;
    }
    Ok(start.elapsed() / N_BENCH as u32)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "estimator-m".into());
    let ep = std::env::args().nth(2).unwrap_or_else(|| "default".into());
    let io = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "no-binding".into());

    let requested_io_binding = match io.as_str() {
        "io-binding" => true,
        "no-binding" => false,
        other => {
            return Err(anyhow!(
                "unknown io mode '{other}': use 'io-binding' or 'no-binding'"
            ));
        }
    };

    let (filename, input_name, output_names, shape): (&str, &str, &[&str], _) = match model.as_str()
    {
        "detector" => (
            RTMDET_NANO_MODEL_NAME,
            RTMDET_INPUT_TENSOR_NAME,
            &[RTMDET_OUTPUT_DETS, RTMDET_OUTPUT_LABELS] as &[&str],
            RTMDET_INPUT_TENSOR_SHAPE,
        ),
        "estimator-m" => (
            RTMPOSE_M_MODEL_NAME,
            RTMPOSE_INPUT_TENSOR_NAME,
            &[RTMPOSE_SIMCC_X_OUTPUT, RTMPOSE_SIMCC_Y_OUTPUT] as &[&str],
            RTMPOSE_INPUT_TENSOR_SHAPE,
        ),
        // Wholebody variants share input shape + tensor names with body
        // RTMPose; only the ONNX file differs.
        "estimator-w-m" => (
            RTMW_M_MODEL_NAME,
            RTMPOSE_INPUT_TENSOR_NAME,
            &[RTMPOSE_SIMCC_X_OUTPUT, RTMPOSE_SIMCC_Y_OUTPUT] as &[&str],
            RTMPOSE_INPUT_TENSOR_SHAPE,
        ),
        "estimator-w-l" => (
            RTMW_L_MODEL_NAME,
            RTMPOSE_INPUT_TENSOR_NAME,
            &[RTMPOSE_SIMCC_X_OUTPUT, RTMPOSE_SIMCC_Y_OUTPUT] as &[&str],
            RTMPOSE_INPUT_TENSOR_SHAPE,
        ),
        other => {
            return Err(anyhow!(
                "unknown model '{other}': use 'estimator-m', 'estimator-w-m', 'estimator-w-l' or 'detector'"
            ));
        }
    };

    let model_path = format!("./models/{filename}");
    println!("model={model}  ep={ep}  io={io}");

    let use_io_binding = resolve_io_mode(&ep, requested_io_binding)?;
    let mut session = build_session(&model_path, &ep)?;

    let avg = if use_io_binding {
        bench_io_binding(&mut session, input_name, output_names, shape)?
    } else {
        bench_run(&mut session, input_name, shape)?
    };

    println!("  avg over {N_BENCH} runs: {avg:?}");
    Ok(())
}
