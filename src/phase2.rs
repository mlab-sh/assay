//! Phase 2 orchestrator: weight inspection. Produces **signals with scores and
//! severities, never verdicts** — a high score means "anomalous, worth a human
//! look", never "malicious". Phase 1 verdicts are untouched.

use std::path::Path;

use crate::dequant::{self, GgmlClass};
use crate::fingerprint::{self, FpInput, Fingerprint};
use crate::formats::{gguf, safetensors};
use crate::numeric::{self, DType};
use crate::profile::{self, ProfilePoint};
use crate::report::{Finding, Severity};
use crate::secrets;
use crate::stats::{self, PerTensorStats, TensorMeta};

pub struct Phase2Opts {
    pub mad_k: f64,
    pub scan_tensor_entropy: bool,
}

#[derive(Default)]
pub struct Phase2Result {
    pub per_tensor: Vec<PerTensorStats>,
    pub layer_profile: Vec<ProfilePoint>,
    pub fingerprint: Option<Fingerprint>,
    pub findings: Vec<Finding>,
}

pub fn run(format: &str, data: &[u8], artifact_path: &Path, opts: &Phase2Opts) -> Phase2Result {
    match format {
        "safetensors" => run_safetensors(data, artifact_path, opts),
        "gguf" => run_gguf(data, artifact_path, opts),
        _ => Phase2Result::default(),
    }
}

fn product(dims: &[u64]) -> u64 {
    let mut p: u128 = 1;
    for &d in dims {
        p = p.saturating_mul(d as u128);
    }
    p.min(u64::MAX as u128) as u64
}

fn nan_inf_finding(s: &PerTensorStats) -> Option<Finding> {
    if s.has_nan_or_inf() {
        Some(
            Finding::new(
                "WEIGHT_NAN_INF",
                Severity::High,
                format!(
                    "tensor '{}' contains {} NaN and {} Inf values — weights should never contain these (corruption or tampering)",
                    s.name,
                    s.nan_count.unwrap_or(0),
                    s.inf_count.unwrap_or(0)
                ),
            )
            .with_evidence(vec![format!("dtype={}, elements={}", s.dtype, s.element_count)]),
        )
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

fn run_safetensors(data: &[u8], artifact_path: &Path, opts: &Phase2Opts) -> Phase2Result {
    let mut res = Phase2Result::default();
    let extract = match safetensors::extract(data) {
        Ok(e) => e,
        Err(_) => return res, // Phase 1 already reports the structural problem
    };

    let mut names = Vec::new();
    for t in &extract.tensors {
        names.push(t.name.clone());
        let element_count = product(&t.shape);
        let meta = TensorMeta {
            name: t.name.clone(),
            dtype_label: t.dtype.clone(),
            shape: t.shape.clone(),
            byte_size: t.len as u64,
            element_count,
        };
        let end = (t.offset + t.len).min(data.len());
        let raw = &data[t.offset.min(data.len())..end];

        let stat = match DType::from_safetensors(&t.dtype) {
            Some(dt) => stats::compute_stats(meta, |f| numeric::stream_values(raw, dt, &mut *f)),
            None => PerTensorStats::deferred(meta),
        };
        if let Some(f) = nan_inf_finding(&stat) {
            res.findings.push(f);
        }
        if opts.scan_tensor_entropy {
            if let Some(f) = tensor_entropy_finding(&t.name, &t.dtype, raw) {
                res.findings.push(f);
            }
        }
        res.per_tensor.push(stat);
    }

    let (points, prof_findings) = profile::build(&res.per_tensor, opts.mad_k);
    res.layer_profile = points;
    res.findings.extend(prof_findings);

    // Secrets: safetensors __metadata__ + sibling config/tokenizer files.
    let meta_pairs: Vec<(String, String)> = extract
        .metadata
        .iter()
        .map(|(k, v)| (format!("metadata.{k}"), v.clone()))
        .collect();
    res.findings
        .extend(secrets::scan(&meta_pairs, artifact_path.parent()));

    // Fingerprint (declared identity + dims from sibling config.json if present).
    let cfg = artifact_path.parent().and_then(read_hf_config);
    let layer_count = names.iter().filter_map(|n| profile::layer_index(n)).max().map(|m| m + 1);
    let (fp, fp_findings) = fingerprint::analyze(&FpInput {
        tensor_names: names,
        declared_arch: cfg.as_ref().and_then(|c| c.model_type.clone()),
        layer_count: cfg.as_ref().and_then(|c| c.layers).or(layer_count),
        hidden: cfg.as_ref().and_then(|c| c.hidden),
        n_heads: cfg.as_ref().and_then(|c| c.heads),
        vocab: cfg.as_ref().and_then(|c| c.vocab),
    });
    res.fingerprint = Some(fp);
    res.findings.extend(fp_findings);

    res
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

fn run_gguf(data: &[u8], artifact_path: &Path, opts: &Phase2Opts) -> Phase2Result {
    let mut res = Phase2Result::default();
    let extract = match gguf::extract(data) {
        Ok(e) => e,
        Err(_) => return res,
    };

    let mut deferred_types: Vec<String> = Vec::new();
    let mut names = Vec::new();

    for t in &extract.tensors {
        names.push(t.name.clone());
        let element_count = product(&t.dims);
        let (type_name, class) = dequant::classify(t.type_id);

        let stat = match class {
            GgmlClass::Plain(dt) => {
                let exact = element_count as usize * dt.size();
                let end = (t.offset + exact).min(data.len());
                let raw = &data[t.offset.min(data.len())..end];
                let meta = TensorMeta {
                    name: t.name.clone(),
                    dtype_label: type_name.to_string(),
                    shape: t.dims.clone(),
                    byte_size: exact as u64,
                    element_count,
                };
                stats::compute_stats(meta, |f| numeric::stream_values(raw, dt, &mut *f))
            }
            GgmlClass::QuantLegacy(kind) => {
                let n_blocks = element_count.div_ceil(32) as usize;
                let exact = n_blocks * kind.block_size();
                let end = (t.offset + exact).min(data.len());
                let raw = &data[t.offset.min(data.len())..end];
                let meta = TensorMeta {
                    name: t.name.clone(),
                    dtype_label: type_name.to_string(),
                    shape: t.dims.clone(),
                    byte_size: exact as u64,
                    element_count,
                };
                stats::compute_stats(meta, |f| {
                    dequant::stream_quant(kind, raw, element_count, &mut *f)
                })
            }
            GgmlClass::Deferred => {
                deferred_types.push(type_name.to_string());
                PerTensorStats::deferred(TensorMeta {
                    name: t.name.clone(),
                    dtype_label: type_name.to_string(),
                    shape: t.dims.clone(),
                    byte_size: t.avail_len as u64,
                    element_count,
                })
            }
        };
        if let Some(f) = nan_inf_finding(&stat) {
            res.findings.push(f);
        }
        res.per_tensor.push(stat);
    }

    if !deferred_types.is_empty() {
        deferred_types.sort();
        deferred_types.dedup();
        res.findings.push(
            Finding::new(
                "STATS_DEFERRED_QUANTIZED",
                Severity::Info,
                "rich stats deferred for k-quant/IQ tensors (dequant not implemented in Phase 2); structural info only",
            )
            .with_evidence(vec![format!("deferred types: {}", deferred_types.join(", "))]),
        );
    }

    let (points, prof_findings) = profile::build(&res.per_tensor, opts.mad_k);
    res.layer_profile = points;
    res.findings.extend(prof_findings);

    // Secrets: GGUF KV string metadata + sibling files.
    res.findings
        .extend(secrets::scan(&extract.kv_strings, artifact_path.parent()));

    // Fingerprint from GGUF metadata.
    let kv = |suffix: &str| -> Option<u64> {
        extract
            .kv_u64
            .iter()
            .find(|(k, _)| k.ends_with(suffix))
            .map(|(_, v)| *v)
    };
    let layer_count = kv(".block_count").or_else(|| {
        names.iter().filter_map(|n| profile::layer_index(n)).max().map(|m| m + 1)
    });
    let (fp, fp_findings) = fingerprint::analyze(&FpInput {
        tensor_names: names,
        declared_arch: extract.architecture.clone(),
        layer_count,
        hidden: kv(".embedding_length"),
        n_heads: kv(".attention.head_count"),
        vocab: kv(".vocab_size"),
    });
    res.fingerprint = Some(fp);
    res.findings.extend(fp_findings);

    res
}

// ---------------------------------------------------------------------------
// Experimental: high-entropy tensor regions (opt-in, low confidence)
// ---------------------------------------------------------------------------

fn tensor_entropy_finding(name: &str, dtype: &str, raw: &[u8]) -> Option<Finding> {
    // Only meaningful for integer/byte tensors — float weights are naturally
    // high-entropy and would false-positive constantly.
    let intish = matches!(dtype, "I8" | "U8" | "BOOL" | "I16" | "I32" | "I64");
    if !intish || raw.len() < 1024 {
        return None;
    }
    let h = secrets::shannon_entropy_bytes(raw);
    if h > 7.95 {
        Some(
            Finding::new(
                "TENSOR_ENTROPY_ANOMALY",
                Severity::Info,
                format!(
                    "[experimental] tensor '{name}' ({dtype}) has near-maximal byte entropy ({h:.2}/8) — \
                     possible packed/smuggled data; high false-positive rate"
                ),
            )
            .with_evidence(vec![format!("entropy={h:.3} bits/byte over {} bytes", raw.len())]),
        )
    } else {
        None
    }
}

// ---------------------------------------------------------------------------

struct HfConfig {
    model_type: Option<String>,
    layers: Option<u64>,
    hidden: Option<u64>,
    heads: Option<u64>,
    vocab: Option<u64>,
}

fn read_hf_config(dir: &Path) -> Option<HfConfig> {
    let p = dir.join("config.json");
    let content = std::fs::read_to_string(&p).ok()?;
    let v: serde_json::Value = serde_json::from_str(&content).ok()?;
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64());
    Some(HfConfig {
        model_type: v.get("model_type").and_then(|x| x.as_str()).map(String::from),
        layers: u("num_hidden_layers").or_else(|| u("n_layer")),
        hidden: u("hidden_size").or_else(|| u("n_embd")),
        heads: u("num_attention_heads").or_else(|| u("n_head")),
        vocab: u("vocab_size"),
    })
}
