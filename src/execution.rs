use crate::shared::execution_providers::register;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    target_os = "windows"
))]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::builder::SessionBuilder;

pub fn with_default_execution_providers(builder: SessionBuilder) -> ort::Result<SessionBuilder> {
    register(
        builder,
        vec![
            #[cfg(any(
                all(target_os = "linux", target_arch = "x86_64"),
                target_os = "windows"
            ))]
            Box::new(CUDAExecutionProvider::default()),
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Box::new(
                CoreMLExecutionProvider::default()
                    .with_compute_units(ort::execution_providers::coreml::ComputeUnits::All)
                    .with_static_input_shapes(true),
            ),
            #[cfg(target_os = "windows")]
            Box::new(DirectMLExecutionProvider::default()),
        ],
    )
}
