use anyhow::Result;
use nalgebra::{Matrix3, Vector3};
use ndarray::ArrayView3;
use tracing::error;

// SimCC frames keypoint localization as 1-D classification along each axis.
// argmax gives the coarse position; the soft-decode below refines to sub-bin
// accuracy with a centre-of-mass over a small window. Matches the reference
// in `mmpose/codecs/simcc_label.py`.
fn decode_axis(logits: &[f32]) -> (f32, f32) {
    // Find argmax. Empty vectors degrade to (0, 0) rather than panicking;
    // upstream code shouldn't pass them, but we never want a panic in a hot
    // inference path.
    let mut peak_idx: usize = 0;
    let mut peak_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > peak_val {
            peak_val = v;
            peak_idx = i;
        }
    }
    if !peak_val.is_finite() {
        return (0.0, 0.0);
    }

    // 3-bin softmax-weighted centroid around the peak. mmpose uses a small
    // window so the refinement stays local even when there's a long tail.
    const WINDOW: i32 = 1;
    let n = logits.len() as i32;
    let mut max_v = f32::NEG_INFINITY;
    for k in -WINDOW..=WINDOW {
        let idx = peak_idx as i32 + k;
        if idx >= 0 && idx < n {
            max_v = max_v.max(logits[idx as usize]);
        }
    }
    let mut weight_sum = 0.0f32;
    let mut weighted_pos = 0.0f32;
    for k in -WINDOW..=WINDOW {
        let idx = peak_idx as i32 + k;
        if idx >= 0 && idx < n {
            let w = (logits[idx as usize] - max_v).exp();
            weight_sum += w;
            weighted_pos += w * idx as f32;
        }
    }

    let refined = if weight_sum > 0.0 {
        weighted_pos / weight_sum
    } else {
        peak_idx as f32
    };

    (refined, peak_val)
}

// Returned positions are in the warped frame (still in pixel units, post-/
// split_ratio); the caller has to map them back through the inverse affine.
// `score` is the weaker of the two axis peaks on the raw scale, as in mmpose's
// `get_simcc_maximum`; an off-frame keypoint is often sharp on one axis only.
pub fn decode_simcc(
    simcc_x: ArrayView3<f32>,
    simcc_y: ArrayView3<f32>,
    simcc_split_ratio: f32,
) -> Result<Vec<(f32, f32, f32)>> {
    if !simcc_split_ratio.is_finite() || simcc_split_ratio <= 0.0 {
        error!(simcc_split_ratio, "simcc_split_ratio must be > 0");
        return Err(anyhow::anyhow!(
            "simcc_split_ratio must be > 0, got {simcc_split_ratio}"
        ));
    }
    let batch_x = simcc_x.shape()[0];
    let batch_y = simcc_y.shape()[0];
    if batch_x != 1 || batch_y != 1 {
        error!(batch_x, batch_y, "decode_simcc expects batch size 1");
        return Err(anyhow::anyhow!(
            "decode_simcc expects batch size 1, got simcc_x={batch_x} simcc_y={batch_y}"
        ));
    }
    let k = simcc_x.shape()[1];
    let k_y = simcc_y.shape()[1];
    if k_y != k {
        error!(k, k_y, "simcc_x and simcc_y keypoint counts differ");
        return Err(anyhow::anyhow!(
            "simcc_x and simcc_y keypoint counts differ: {k} vs {k_y}"
        ));
    }

    let mut out = Vec::with_capacity(k);
    for keypoint in 0..k {
        let x_row = simcc_x.slice(ndarray::s![0, keypoint, ..]);
        let y_row = simcc_y.slice(ndarray::s![0, keypoint, ..]);
        // Row views over the last axis of a C-order array are always contiguous,
        // so as_slice() is guaranteed to succeed here.
        let x_slice = x_row
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("non-contiguous simcc_x row for keypoint {keypoint}"))?;
        let y_slice = y_row
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("non-contiguous simcc_y row for keypoint {keypoint}"))?;
        let (x_bin, x_score) = decode_axis(x_slice);
        let (y_bin, y_score) = decode_axis(y_slice);
        let x_pixel = x_bin / simcc_split_ratio;
        let y_pixel = y_bin / simcc_split_ratio;
        let score = x_score.min(y_score);
        out.push((x_pixel, y_pixel, score));
    }
    Ok(out)
}

pub fn apply_inverse_affine(forward: &Matrix3<f32>, x: f32, y: f32) -> Option<(f32, f32)> {
    let inv = forward.try_inverse()?;
    let p = inv * Vector3::new(x, y, 1.0);
    Some((p.x, p.y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;
    use parameterized::parameterized;

    fn one_hot(bins: usize, peak: usize) -> Vec<f32> {
        // Sharp logits so the soft-decode collapses to the peak index.
        let mut v = vec![-50.0_f32; bins];
        if peak < bins {
            v[peak] = 10.0;
        }
        v
    }

    #[test]
    fn test_decode_axis_picks_argmax_with_sharp_logits() {
        // Arrange
        let logits = one_hot(20, 7);

        // Act
        let (pos, score) = decode_axis(&logits);

        // Assert
        assert_abs_diff_eq!(pos, 7.0, epsilon = 1e-3);
        assert_abs_diff_eq!(score, 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_decode_axis_subbin_refinement_between_neighbours() {
        // Arrange — symmetric peak between bins 7 and 8 should land near 7.5
        // after the soft-decode.
        let mut logits = vec![-10.0_f32; 20];
        logits[7] = 5.0;
        logits[8] = 5.0;

        // Act
        let (pos, _) = decode_axis(&logits);

        // Assert — argmax breaks the tie at 7, but the centroid pulls toward 7.5.
        assert!(
            (7.0..=8.0).contains(&pos),
            "refined position must lie between the two peaks, got {pos}"
        );
        assert!(pos > 7.05, "should pull away from raw argmax, got {pos}");
    }

    #[test]
    fn test_decode_axis_handles_empty_input() {
        // Arrange
        let logits: Vec<f32> = Vec::new();

        // Act
        let (pos, score) = decode_axis(&logits);

        // Assert — degrade to zero rather than panic.
        assert_eq!(pos, 0.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_decode_axis_handles_peak_at_boundary() {
        // Arrange — peak at idx 0; the window clipping shouldn't break decode.
        let logits = one_hot(20, 0);

        // Act
        let (pos, _) = decode_axis(&logits);

        // Assert
        assert_abs_diff_eq!(pos, 0.0, epsilon = 1e-3);
    }

    fn make_simcc(
        k: usize,
        x_bins: usize,
        y_bins: usize,
        peaks: &[(usize, usize)],
    ) -> (Array3<f32>, Array3<f32>) {
        let mut x = Array3::<f32>::from_elem((1, k, x_bins), -50.0);
        let mut y = Array3::<f32>::from_elem((1, k, y_bins), -50.0);
        for (kp, (px, py)) in peaks.iter().enumerate() {
            x[[0, kp, *px]] = 10.0;
            y[[0, kp, *py]] = 10.0;
        }
        (x, y)
    }

    #[parameterized(
        k = { 1, 17, 133 },
        split_ratio = { 1.0_f32, 2.0_f32, 3.0_f32 }
    )]
    fn test_decode_simcc_recovers_peak_locations(k: usize, split_ratio: f32) {
        // Arrange — each keypoint i peaks at bin (i % w, i % h). With sharp
        // logits the decoder must recover those positions / split_ratio.
        let w = 192_usize * split_ratio as usize;
        let h = 256_usize * split_ratio as usize;
        let peaks: Vec<(usize, usize)> = (0..k).map(|i| (i % w, i % h)).collect();
        let (x, y) = make_simcc(k, w, h, &peaks);

        // Act
        let decoded = decode_simcc(x.view(), y.view(), split_ratio).unwrap();

        // Assert
        assert_eq!(decoded.len(), k);
        for (i, (dx, dy, _)) in decoded.iter().enumerate() {
            assert_abs_diff_eq!(*dx, (i % w) as f32 / split_ratio, epsilon = 1e-3);
            assert_abs_diff_eq!(*dy, (i % h) as f32 / split_ratio, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_decode_simcc_scores_keypoint_by_its_weaker_axis() {
        // Arrange — single keypoint, x-peak is sharper than y. A keypoint
        // pinned down on one axis only (e.g. a foot below the frame edge) must
        // not count as confident, so the score is the weaker peak (4.0).
        let mut x = Array3::<f32>::from_elem((1, 1, 10), -50.0);
        let mut y = Array3::<f32>::from_elem((1, 1, 10), -50.0);
        x[[0, 0, 3]] = 10.0;
        y[[0, 0, 7]] = 4.0;

        // Act
        let decoded = decode_simcc(x.view(), y.view(), 1.0).unwrap();

        // Assert
        assert_eq!(decoded.len(), 1);
        assert_abs_diff_eq!(decoded[0].2, 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_inverse_affine_round_trip() {
        // Arrange — a simple scale+translate forward matrix.
        let forward = Matrix3::new(2.0, 0.0, 10.0, 0.0, 2.0, 20.0, 0.0, 0.0, 1.0);
        let (src_x, src_y) = (5.0, 7.0);
        let warped = forward * Vector3::new(src_x, src_y, 1.0);

        // Act
        let (back_x, back_y) = apply_inverse_affine(&forward, warped.x, warped.y).unwrap();

        // Assert
        assert_abs_diff_eq!(back_x, src_x, epsilon = 1e-5);
        assert_abs_diff_eq!(back_y, src_y, epsilon = 1e-5);
    }
}
