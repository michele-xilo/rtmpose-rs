use ort::execution_providers::ExecutionProvider;
use ort::session::builder::SessionBuilder;
use tracing::{info, warn};

pub type Providers = Vec<Box<dyn ExecutionProvider>>;

// ort registers execution providers silently and falls back to CPU when one
// fails to load (a missing cuDNN 9 on Linux is the usual cause), so registration
// is done one provider at a time to report which ones actually took. Successful
// registration only means the provider is available to the session; which nodes
// it ends up running is decided later by ORT's placement, which it logs itself.
pub fn register(mut builder: SessionBuilder, providers: Providers) -> ort::Result<SessionBuilder> {
    let mut active = Vec::new();
    for provider in providers {
        match provider.register(&mut builder) {
            Ok(()) => active.push(provider.name()),
            Err(e) => warn!(
                "Execution provider {} could not be registered: {e}",
                provider.name()
            ),
        }
    }
    info!("{}", describe(&active));
    Ok(builder)
}

fn describe(active: &[&str]) -> String {
    if active.is_empty() {
        "Inference running on CPU".to_string()
    } else {
        format!("Execution providers available: {}", active.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_cpu_only_when_nothing_registered() {
        // Arrange
        let active: [&str; 0] = [];

        // Act
        let message = describe(&active);

        // Assert
        assert_eq!(message, "Inference running on CPU");
    }

    #[test]
    fn describes_the_single_registered_provider() {
        // Arrange
        let active = ["CUDAExecutionProvider"];

        // Act
        let message = describe(&active);

        // Assert
        assert_eq!(
            message,
            "Execution providers available: CUDAExecutionProvider"
        );
    }

    #[test]
    fn describes_every_registered_provider_in_priority_order() {
        // Arrange
        let active = ["CUDAExecutionProvider", "DmlExecutionProvider"];

        // Act
        let message = describe(&active);

        // Assert
        assert_eq!(
            message,
            "Execution providers available: CUDAExecutionProvider, DmlExecutionProvider"
        );
    }
}
