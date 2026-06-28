//! 2b — layer profile + robust (median/MAD) anomaly detection.
//!
//! Tensors are bucketed into layers by parsing their names. For each layer we
//! aggregate the 2a per-tensor stats into a profile point, then flag layers
//! whose metric deviates from the cross-layer **median** by more than `k`
//! **MADs** (median absolute deviation). Robust stats are essential: a single
//! tampered layer must not inflate the threshold enough to hide itself, which a
//! mean/std rule would allow.

pub mod render;

use serde::Serialize;

use crate::report::{Finding, Severity};
use crate::stats::PerTensorStats;

#[derive(Debug, Clone, Serialize)]
pub struct Anomaly {
    pub metric: String,
    pub mads: f64,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfilePoint {
    pub layer: u64,
    pub l2: f64,
    pub mean_kurtosis: f64,
    pub max_abs: f64,
    pub params: u64,
    pub sparsity: f64,
    pub anomaly: Option<Anomaly>,
}

/// Parse a layer index from a tensor name across common naming schemes
/// (`model.layers.N`, `transformer.h.N`, `h.N`, `blk.N`, `…layer.N…`).
pub fn layer_index(name: &str) -> Option<u64> {
    let toks: Vec<&str> = name.split('.').collect();
    const KEYS: [&str; 5] = ["layers", "h", "blk", "block", "layer"];
    for i in 0..toks.len().saturating_sub(1) {
        if KEYS.contains(&toks[i]) {
            if let Ok(n) = toks[i + 1].parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

struct Acc {
    sum_sq_l2: f64,
    kurt_sum: f64,
    kurt_n: u64,
    max_abs: f64,
    params: u64,
    zeros: f64,
    elems: f64,
}

impl Default for Acc {
    fn default() -> Self {
        Acc {
            sum_sq_l2: 0.0,
            kurt_sum: 0.0,
            kurt_n: 0,
            max_abs: 0.0,
            params: 0,
            zeros: 0.0,
            elems: 0.0,
        }
    }
}

/// Build the layer profile and emit `WEIGHT_OUTLIER_LAYER` findings.
pub fn build(per_tensor: &[PerTensorStats], mad_k: f64) -> (Vec<ProfilePoint>, Vec<Finding>) {
    use std::collections::BTreeMap;
    let mut layers: BTreeMap<u64, Acc> = BTreeMap::new();

    for t in per_tensor {
        let Some(idx) = layer_index(&t.name) else {
            continue;
        };
        let a = layers.entry(idx).or_default();
        a.params += t.element_count;
        a.elems += t.element_count as f64;
        if let Some(l2) = t.l2_norm {
            a.sum_sq_l2 += l2 * l2;
        }
        if let Some(k) = t.kurtosis {
            a.kurt_sum += k;
            a.kurt_n += 1;
        }
        if let Some(ma) = t.max_abs {
            if ma > a.max_abs {
                a.max_abs = ma;
            }
        }
        if let (Some(sp), ec) = (t.sparsity, t.element_count) {
            a.zeros += sp * ec as f64;
        }
    }

    let mut points: Vec<ProfilePoint> = layers
        .into_iter()
        .map(|(layer, a)| ProfilePoint {
            layer,
            l2: a.sum_sq_l2.sqrt(),
            mean_kurtosis: if a.kurt_n > 0 {
                a.kurt_sum / a.kurt_n as f64
            } else {
                0.0
            },
            max_abs: a.max_abs,
            params: a.params,
            sparsity: if a.elems > 0.0 { a.zeros / a.elems } else { 0.0 },
            anomaly: None,
        })
        .collect();

    let findings = detect_anomalies(&mut points, mad_k);
    (points, findings)
}

const METRICS: [&str; 3] = ["l2", "mean_kurtosis", "max_abs"];

fn metric_value(p: &ProfilePoint, m: &str) -> f64 {
    match m {
        "l2" => p.l2,
        "mean_kurtosis" => p.mean_kurtosis,
        "max_abs" => p.max_abs,
        _ => 0.0,
    }
}

fn detect_anomalies(points: &mut [ProfilePoint], mad_k: f64) -> Vec<Finding> {
    let mut findings = Vec::new();
    // MAD is unreliable with too few samples.
    if points.len() < 4 {
        return findings;
    }

    // Precompute (median, mad) per metric.
    let mut stats = Vec::new();
    for m in METRICS {
        let vals: Vec<f64> = points.iter().map(|p| metric_value(p, m)).collect();
        let med = median(&vals);
        let dev: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
        let mut mad = median(&dev);
        if mad <= f64::EPSILON {
            // Fall back to mean absolute deviation when MAD collapses.
            let mean_abs = dev.iter().sum::<f64>() / dev.len() as f64;
            mad = mean_abs;
        }
        stats.push((med, mad));
    }

    for p in points.iter_mut() {
        let mut best: Option<Anomaly> = None;
        for (mi, m) in METRICS.iter().enumerate() {
            let (med, mad) = stats[mi];
            if mad <= f64::EPSILON {
                continue;
            }
            let mads = (metric_value(p, m) - med).abs() / mad;
            if mads > mad_k && best.as_ref().map(|b| mads > b.mads).unwrap_or(true) {
                let severity = if mads >= 2.0 * mad_k { "medium" } else { "low" };
                best = Some(Anomaly {
                    metric: (*m).to_string(),
                    mads,
                    severity: severity.to_string(),
                });
            }
        }
        if let Some(a) = best {
            let sev = if a.severity == "medium" {
                Severity::Medium
            } else {
                Severity::Low
            };
            findings.push(
                Finding::new(
                    "WEIGHT_OUTLIER_LAYER",
                    sev,
                    format!(
                        "layer {} is anomalous on {} ({:.1} MADs from the cross-layer median) — \
                         worth a human look, not a verdict",
                        p.layer, a.metric, a.mads
                    ),
                )
                .with_evidence(vec![format!(
                    "metric={}, value={:.4}, mads={:.2}",
                    a.metric,
                    metric_value(p, &a.metric),
                    a.mads
                )]),
            );
            p.anomaly = Some(a);
        }
    }

    findings
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_layer_schemes() {
        assert_eq!(layer_index("model.layers.7.mlp.up_proj.weight"), Some(7));
        assert_eq!(layer_index("transformer.h.3.attn.c_attn.weight"), Some(3));
        assert_eq!(layer_index("h.0.ln_1.weight"), Some(0));
        assert_eq!(layer_index("blk.12.attn_k.weight"), Some(12));
        assert_eq!(layer_index("wte.weight"), None);
    }
}
