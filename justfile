# Recipes are tagged with [group(...)]; `just --list` prints them by section.
# Requires just >= 1.27.

default:
    @just --list

# ---------------------------------------------------------------------------
# Checks: lint, format, test, audit
# ---------------------------------------------------------------------------

[group('check')]
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings && \
    cargo fmt --all -- --check && \
    cargo machete

[group('check')]
format:
    cargo fmt --all && \
    cargo fix --allow-dirty --workspace

[group('check')]
test:
    cargo nextest run --workspace --all-features

# Runs the tests that need ONNX weights in ./models and a photo with at least
# one person in RTMPOSE_TEST_IMAGE (defaults to assets/test_image.jpg).
[group('check')]
test-models:
    cargo nextest run --all-features --run-ignored only

[group('check')]
audit:
    cargo audit

[group('check')]
licenses:
    cargo deny check licenses

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

[group('build')]
build:
    cargo build

[group('build')]
release:
    cargo build --release

[group('build')]
clean:
    cargo clean

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

[group('bench')]
bench:
    cargo bench --bench pose_rtm_benchmarks

# Inference latency per execution provider; see examples/ep_bench.rs for the
# models, EPs and IO modes.
[group('bench')]
bench-ep MODEL="estimator-m" EP="default" IO="no-binding":
    RUST_LOG=info,ort=error cargo run --release --example ep_bench -- {{ MODEL }} {{ EP }} {{ IO }}

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

[group('example')]
example-pose-rtm IMAGE:
    RUST_LOG=info,ort=error cargo run --release --example pose_rtm -- {{ IMAGE }}

[group('example')]
example-pose-rtmw IMAGE:
    RUST_LOG=info,ort=error cargo run --release --example pose_rtmw -- {{ IMAGE }}

[group('example')]
camera-pose-rtm:
    RUST_LOG=demo=info,rtmpose=debug,ort=error cargo run --release -p demo --bin camera-pose-rtm

[group('example')]
camera-pose-rtmw:
    RUST_LOG=demo=info,rtmpose=debug,ort=error cargo run --release -p demo --bin camera-pose-rtmw

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Pulls this repo's ONNX models from Hugging Face into models/.
[group('models')]
get-models:
    hf download michele-xilo/rtmpose-rs --local-dir models --include "*.onnx"
