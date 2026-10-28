#[macro_export]
macro_rules! time_it {
    ($label:expr_2021, $expr:expr_2021) => {{
        let start = std::time::Instant::now();
        let result = $expr;
        let duration = start.elapsed();
        tracing::debug!("[{}] Execution time: {:?}", $label, duration);
        result
    }};
}
