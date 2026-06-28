//! 2a — per-tensor streaming statistics.
//!
//! Given a way to stream a tensor's values (a `run_pass` closure that can be
//! invoked more than once), compute integrity + distribution stats in O(1)
//! memory. The closure indirection lets the same driver serve raw safetensors
//! bytes and on-the-fly dequantized GGUF blocks identically.

pub mod moments;

use serde::Serialize;

use moments::Moments;

/// Identifying metadata for a tensor, known before we read values.
pub struct TensorMeta {
    pub name: String,
    pub dtype_label: String,
    pub shape: Vec<u64>,
    pub byte_size: u64,
    pub element_count: u64,
}

/// Whether a tensor got real stats or was deferred (e.g. k-quant GGUF).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Full,
    DeferredQuantized,
}

impl Quality {
    fn as_str(self) -> &'static str {
        match self {
            Quality::Full => "full",
            Quality::DeferredQuantized => "deferred_quantized",
        }
    }
}

/// Serializable 2a result for one tensor. Numeric fields are absent (null) when
/// stats were deferred, so the JSON never implies a value we didn't compute.
#[derive(Debug, Clone, Serialize)]
pub struct PerTensorStats {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub byte_size: u64,
    pub quality: &'static str,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nan_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inf_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub l2_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kurtosis: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparsity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outlier_mass: Option<f64>,
}

impl PerTensorStats {
    /// Build a deferred (no real stats) result — used for k-quant GGUF tensors.
    pub fn deferred(meta: TensorMeta) -> Self {
        PerTensorStats {
            name: meta.name,
            dtype: meta.dtype_label,
            shape: meta.shape,
            element_count: meta.element_count,
            byte_size: meta.byte_size,
            quality: Quality::DeferredQuantized.as_str(),
            nan_count: None,
            inf_count: None,
            min: None,
            max: None,
            max_abs: None,
            mean: None,
            std: None,
            l2_norm: None,
            rms: None,
            kurtosis: None,
            sparsity: None,
            outlier_mass: None,
        }
    }

    pub fn has_nan_or_inf(&self) -> bool {
        self.nan_count.unwrap_or(0) > 0 || self.inf_count.unwrap_or(0) > 0
    }
}

/// Compute full per-tensor stats. `run_pass(&mut f)` must stream every value of
/// the tensor through `f`; it is called twice (moments, then a 6σ outlier pass)
/// and must therefore be deterministic and side-effect free.
pub fn compute_stats<R>(meta: TensorMeta, run_pass: R) -> PerTensorStats
where
    R: Fn(&mut dyn FnMut(f64)),
{
    let mut mom = Moments::new();
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;
    let mut zeros = 0u64;
    let mut finite_n = 0u64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs = 0.0f64;

    run_pass(&mut |x| {
        if x.is_nan() {
            nan_count += 1;
            return;
        }
        if x.is_infinite() {
            inf_count += 1;
            return;
        }
        finite_n += 1;
        if x == 0.0 {
            zeros += 1;
        }
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
        mom.push(x);
    });

    if finite_n == 0 {
        // Nothing finite to summarize, but still report integrity counts.
        return PerTensorStats {
            name: meta.name,
            dtype: meta.dtype_label,
            shape: meta.shape,
            element_count: meta.element_count,
            byte_size: meta.byte_size,
            quality: Quality::Full.as_str(),
            nan_count: Some(nan_count),
            inf_count: Some(inf_count),
            min: None,
            max: None,
            max_abs: None,
            mean: None,
            std: None,
            l2_norm: None,
            rms: None,
            kurtosis: None,
            sparsity: Some(0.0),
            outlier_mass: None,
        };
    }

    let std = mom.std();
    let l2 = mom.sum_squares().max(0.0).sqrt();
    let rms = l2 / (finite_n as f64).sqrt();
    let sparsity = zeros as f64 / meta.element_count.max(1) as f64;

    // Second pass: 6σ outlier mass (needs the now-known std).
    let mut outliers = 0u64;
    if std > 0.0 {
        let thr = 6.0 * std;
        run_pass(&mut |x| {
            if x.is_finite() && x.abs() > thr {
                outliers += 1;
            }
        });
    }
    let outlier_mass = outliers as f64 / finite_n as f64;

    PerTensorStats {
        name: meta.name,
        dtype: meta.dtype_label,
        shape: meta.shape,
        element_count: meta.element_count,
        byte_size: meta.byte_size,
        quality: Quality::Full.as_str(),
        nan_count: Some(nan_count),
        inf_count: Some(inf_count),
        min: Some(min),
        max: Some(max),
        max_abs: Some(max_abs),
        mean: Some(mom.mean()),
        std: Some(std),
        l2_norm: Some(l2),
        rms: Some(rms),
        kurtosis: Some(mom.excess_kurtosis()),
        sparsity: Some(sparsity),
        outlier_mass: Some(outlier_mass),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{stream_values, DType};

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut b = Vec::new();
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    #[test]
    fn basic_stats() {
        let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0]);
        let meta = TensorMeta {
            name: "t".into(),
            dtype_label: "F32".into(),
            shape: vec![5],
            byte_size: bytes.len() as u64,
            element_count: 5,
        };
        let s = compute_stats(meta, |f| stream_values(&bytes, DType::F32, &mut *f));
        assert_eq!(s.nan_count, Some(0));
        assert_eq!(s.element_count, 5);
        assert!((s.mean.unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(s.sparsity, Some(0.2)); // one exact zero of five
        assert_eq!(s.max, Some(4.0));
    }

    #[test]
    fn nan_inf_counted_not_poisoning() {
        let bytes = f32_bytes(&[1.0, f32::NAN, 2.0, f32::INFINITY, 3.0]);
        let meta = TensorMeta {
            name: "t".into(),
            dtype_label: "F32".into(),
            shape: vec![5],
            byte_size: bytes.len() as u64,
            element_count: 5,
        };
        let s = compute_stats(meta, |f| stream_values(&bytes, DType::F32, &mut *f));
        assert_eq!(s.nan_count, Some(1));
        assert_eq!(s.inf_count, Some(1));
        assert!(s.has_nan_or_inf());
        assert!((s.mean.unwrap() - 2.0).abs() < 1e-9); // mean over 1,2,3
    }
}
