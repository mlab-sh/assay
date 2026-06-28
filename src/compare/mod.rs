//! `compare SUBJECT BASELINE` — differential weight analysis.
//!
//! The diagnostic signal for tampering is not a weight's absolute value but how
//! the subject **differs from a known-good reference** of the same architecture.
//! Identical models → silence. A localized tamper → it lights up exactly where
//! the tamper is, and nowhere else. Like Phase 2 these are **signals, not
//! verdicts** — the one near-verdict-grade exception is structural divergence
//! (a tensor present in one model but not the other, or a shape mismatch).
//!
//! Streaming: both files are mmap'd and processed one matched tensor pair at a
//! time (lockstep pull iterators), so neither full model is held in RAM.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::dequant::{self, GgmlClass};
use crate::fingerprint::{self, FpInput};
use crate::format::{self, Format};
use crate::formats::{gguf, safetensors};
use crate::mapio::map_file;
use crate::report::{Finding, Severity};
use crate::stats::moments::Moments;
use crate::values::ValueIter;

pub struct CompareOpts {
    pub mad_k: f64,
    pub epsilon: f64,
    pub force: bool,
}

// ---- decoding spec for one tensor (borrows the mmap) ----
enum Decoder {
    St(String),
    Gg { type_id: u32, n: u64 },
}

struct T<'a> {
    /// Original on-disk tensor name (what we print).
    name: String,
    shape: Vec<u64>,
    raw: &'a [u8],
    dec: Decoder,
}

/// Wrapper prefixes that the same architecture may or may not carry depending on
/// how it was serialized. Stripped symmetrically before alignment so a "bare"
/// model and a `*LMHeadModel`-saved fine-tune line up.
const WRAPPER_PREFIXES: [&str; 4] = ["transformer.", "model.", "module.", "_orig_mod."];

/// Canonicalize a tensor name by repeatedly stripping known wrapper prefixes
/// (they can stack, e.g. `module._orig_mod.transformer.…`).
fn canonical(name: &str) -> String {
    let mut s = name;
    'outer: loop {
        for p in WRAPPER_PREFIXES {
            if let Some(rest) = s.strip_prefix(p) {
                s = rest;
                continue 'outer;
            }
        }
        break;
    }
    s.to_string()
}

/// Known weight-tying relations: a missing tensor explained by being tied to a
/// counterpart is a serialization convention, not a divergence. Keyed on
/// canonical names.
const TIE_ALIASES: [(&str, &[&str]); 1] =
    [("lm_head.weight", &["wte.weight", "embed_tokens.weight"])];

fn tied_counterparts(canon: &str) -> Option<&'static [&'static str]> {
    TIE_ALIASES
        .iter()
        .find(|(k, _)| *k == canon)
        .map(|(_, v)| *v)
}

impl<'a> T<'a> {
    fn iter(&self) -> Option<ValueIter<'a>> {
        match &self.dec {
            Decoder::St(dtype) => ValueIter::safetensors(self.raw, dtype),
            Decoder::Gg { type_id, n } => ValueIter::gguf(self.raw, *type_id, *n),
        }
    }
    fn deferred(&self) -> bool {
        self.iter().is_none()
    }
}

// ---- serializable report ----

#[derive(Serialize)]
pub struct ArchInfo {
    pub subject: String,
    pub baseline: String,
    #[serde(rename = "match")]
    pub matched: bool,
}

#[derive(Serialize)]
pub struct StructuralDivergence {
    pub name: String,
    pub kind: String,
    pub detail: String,
}

#[derive(Serialize)]
pub struct TensorDrift {
    pub name: String,
    pub cosine: f64,
    pub rel_l2: f64,
    pub changed_frac: f64,
    pub delta_max_abs: f64,
    pub delta_rms: f64,
    pub delta_std: f64,
    pub delta_kurtosis: f64,
    #[serde(skip)]
    diff2: f64,
    #[serde(skip)]
    bb: f64,
}

#[derive(Serialize)]
pub struct DriftAnomaly {
    pub mads: f64,
    pub severity: String,
}

#[derive(Serialize)]
pub struct LayerDrift {
    pub layer: u64,
    pub rel_l2: f64,
    pub anomaly: Option<DriftAnomaly>,
}

#[derive(Serialize)]
pub struct Summary {
    pub matched: u64,
    pub identical_fraction: f64,
    pub structural: u64,
    pub worst_drift: f64,
}

#[derive(Serialize)]
pub struct CompareReport {
    pub subject: String,
    pub baseline: String,
    pub arch: ArchInfo,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub normalization: Vec<String>,
    pub structural_divergences: Vec<StructuralDivergence>,
    pub tensor_drift: Vec<TensorDrift>,
    pub layer_drift: Vec<LayerDrift>,
    pub findings: Vec<Finding>,
    pub summary: Summary,
}

impl CompareReport {
    pub fn max_severity(&self) -> Option<Severity> {
        self.findings.iter().map(|f| f.severity).max()
    }
    /// Exit code: 3 internal (unused here) > 1 findings >= threshold > 0.
    pub fn exit_code(&self, fail_on: Severity) -> i32 {
        match self.max_severity() {
            Some(s) if s >= fail_on => 1,
            _ => 0,
        }
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

fn product(dims: &[u64]) -> u64 {
    let mut p: u128 = 1;
    for &d in dims {
        p = p.saturating_mul(d as u128);
    }
    p.min(u64::MAX as u128) as u64
}

/// Build a canonical-name→tensor map for a model, slicing exact tensor byte
/// ranges. Keys are canonicalized; each `T` retains its original name.
/// Returns the map plus the number of keys whose name was normalized.
fn build_tensors<'a>(fmt: Format, data: &'a [u8]) -> Result<(BTreeMap<String, T<'a>>, usize), String> {
    let mut map = BTreeMap::new();
    let mut stripped = 0usize;
    let mut insert = |name: String, shape, raw, dec, map: &mut BTreeMap<String, T<'a>>| {
        let canon = canonical(&name);
        if canon != name {
            stripped += 1;
        }
        map.insert(canon, T { name, shape, raw, dec });
    };
    match fmt {
        Format::Safetensors => {
            let ex = safetensors::extract(data)?;
            for t in ex.tensors {
                let raw = &data[t.offset.min(data.len())..(t.offset + t.len).min(data.len())];
                insert(t.name, t.shape, raw, Decoder::St(t.dtype), &mut map);
            }
        }
        Format::Gguf => {
            let ex = gguf::extract(data)?;
            for t in ex.tensors {
                let n = product(&t.dims);
                let exact = match dequant::classify(t.type_id).1 {
                    GgmlClass::Plain(dt) => n as usize * dt.size(),
                    GgmlClass::QuantLegacy(kind) => n.div_ceil(32) as usize * kind.block_size(),
                    GgmlClass::Deferred => t.avail_len,
                };
                let end = (t.offset + exact).min(data.len());
                let raw = &data[t.offset.min(data.len())..end];
                insert(t.name, t.dims, raw, Decoder::Gg { type_id: t.type_id, n }, &mut map);
            }
        }
        _ => return Err("compare supports safetensors and GGUF only".into()),
    }
    Ok((map, stripped))
}

fn fingerprint_of(fmt: Format, data: &[u8], names: &[String]) -> fingerprint::Fingerprint {
    let declared = if fmt == Format::Gguf {
        gguf::extract(data).ok().and_then(|e| e.architecture)
    } else {
        None
    };
    let layer_count = names
        .iter()
        .filter_map(|n| crate::profile::layer_index(n))
        .max()
        .map(|m| m + 1);
    fingerprint::analyze(&FpInput {
        tensor_names: names.to_vec(),
        declared_arch: declared,
        layer_count,
        hidden: None,
        n_heads: None,
        vocab: None,
    })
    .0
}

/// Resolve a path to a single weight artifact. Accepts a file directly, or a
/// directory containing exactly one (or a `model.safetensors`) artifact.
fn resolve_model(p: &Path) -> Result<PathBuf, String> {
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if !p.is_dir() {
        return Err(format!("path not found: {}", p.display()));
    }
    let mut st = Vec::new();
    let mut gg = Vec::new();
    for e in std::fs::read_dir(p).map_err(|e| e.to_string())?.flatten() {
        let pa = e.path();
        match pa.extension().and_then(|x| x.to_str()) {
            Some("safetensors") => st.push(pa),
            Some("gguf") => gg.push(pa),
            _ => {}
        }
    }
    st.sort();
    gg.sort();
    if let Some(m) = st
        .iter()
        .find(|x| x.file_name().and_then(|n| n.to_str()) == Some("model.safetensors"))
    {
        return Ok(m.clone());
    }
    match (st.len(), gg.len()) {
        (1, _) => Ok(st[0].clone()),
        (0, 1) => Ok(gg[0].clone()),
        (0, 0) => Err(format!("no safetensors or GGUF artifact in {}", p.display())),
        _ => Err(format!(
            "{} contains multiple weight files (e.g. sharded); point compare at a single file",
            p.display()
        )),
    }
}

pub fn run(subject: &Path, baseline: &Path, opts: &CompareOpts) -> CompareReport {
    let subject = match resolve_model(subject) {
        Ok(p) => p,
        Err(e) => return error_report(subject, baseline, &e),
    };
    let baseline = match resolve_model(baseline) {
        Ok(p) => p,
        Err(e) => return error_report(&subject, baseline, &e),
    };
    let subject = subject.as_path();
    let baseline = baseline.as_path();

    let s_bytes = map_file(subject);
    let b_bytes = map_file(baseline);

    // Errors / unreadable -> minimal report with a high finding.
    let (s_data, b_data) = match (&s_bytes, &b_bytes) {
        (Ok(s), Ok(b)) => (&**s, &**b),
        _ => return error_report(subject, baseline, "could not read subject or baseline file"),
    };

    let s_fmt = detect(subject, s_data);
    let b_fmt = detect(baseline, b_data);
    if s_fmt != b_fmt || !matches!(s_fmt, Format::Safetensors | Format::Gguf) {
        return error_report(
            subject,
            baseline,
            "compare needs two artifacts of the same supported format (safetensors or GGUF)",
        );
    }

    let (s_map, s_stripped) = match build_tensors(s_fmt, s_data) {
        Ok(m) => m,
        Err(e) => return error_report(subject, baseline, &e),
    };
    let (b_map, b_stripped) = match build_tensors(b_fmt, b_data) {
        Ok(m) => m,
        Err(e) => return error_report(subject, baseline, &e),
    };

    // Audit trail: what canonicalization normalized away.
    let mut normalization = Vec::new();
    if s_stripped > 0 {
        normalization.push(format!("stripped wrapper prefix from {s_stripped} subject tensor name(s)"));
    }
    if b_stripped > 0 {
        normalization.push(format!("stripped wrapper prefix from {b_stripped} baseline tensor name(s)"));
    }

    // Fingerprint on canonical names so naming convention doesn't skew arch.
    let s_names: Vec<String> = s_map.keys().cloned().collect();
    let b_names: Vec<String> = b_map.keys().cloned().collect();
    let fp_s = fingerprint_of(s_fmt, s_data, &s_names);
    let fp_b = fingerprint_of(b_fmt, b_data, &b_names);
    let arch_label = |f: &fingerprint::Fingerprint| {
        f.detected_family.clone().unwrap_or_else(|| f.scheme.clone())
    };
    let arch_match = fp_s.scheme == fp_b.scheme && fp_s.layer_count == fp_b.layer_count;
    let arch = ArchInfo {
        subject: arch_label(&fp_s),
        baseline: arch_label(&fp_b),
        matched: arch_match,
    };

    let mut findings = Vec::new();

    // Arch guard: refuse to emit drift across architectures unless --force.
    if !arch_match && !opts.force {
        findings.push(
            Finding::new(
                "ARCH_MISMATCH",
                Severity::High,
                format!(
                    "subject ({}, {} layers) and baseline ({}, {} layers) are different architectures — \
                     cross-architecture weight drift is meaningless; pass --force to compare anyway",
                    arch.subject,
                    fp_s.layer_count.unwrap_or(0),
                    arch.baseline,
                    fp_b.layer_count.unwrap_or(0)
                ),
            ),
        );
        return CompareReport {
            subject: subject.display().to_string(),
            baseline: baseline.display().to_string(),
            arch,
            normalization: normalization.clone(),
            structural_divergences: Vec::new(),
            tensor_drift: Vec::new(),
            layer_drift: Vec::new(),
            findings,
            summary: Summary {
                matched: 0,
                identical_fraction: 0.0,
                structural: 0,
                worst_drift: 0.0,
            },
        };
    }
    if !arch_match && opts.force {
        findings.push(Finding::new(
            "ARCH_MISMATCH",
            Severity::Medium,
            "architectures differ; --force given, drift scores below are UNRELIABLE",
        ));
    }

    // ---- alignment + drift ----
    let mut structural = Vec::new();
    let mut tensor_drift: Vec<TensorDrift> = Vec::new();
    let mut deferred_pairs = 0u64;

    let all_keys: std::collections::BTreeSet<&String> =
        s_map.keys().chain(b_map.keys()).collect();

    for key in all_keys {
        match (s_map.get(key), b_map.get(key)) {
            (Some(s), Some(b)) => {
                if s.shape != b.shape {
                    structural.push(StructuralDivergence {
                        name: s.name.clone(),
                        kind: "shape_mismatch".into(),
                        detail: format!("subject {:?} vs baseline {:?}", s.shape, b.shape),
                    });
                } else if s.deferred() || b.deferred() {
                    deferred_pairs += 1;
                } else if let (Some(si), Some(bi)) = (s.iter(), b.iter()) {
                    tensor_drift.push(drift(&s.name, si, bi, opts.epsilon));
                }
            }
            // Present only on one side: a tie explains it (info), else divergence.
            (Some(s), None) => match tie_finding(key, &s_map, s, opts.epsilon) {
                Some(f) => findings.push(f),
                None => structural.push(StructuralDivergence {
                    name: s.name.clone(),
                    kind: "only_in_subject".into(),
                    detail: "tensor present in subject but not baseline".into(),
                }),
            },
            (None, Some(b)) => match tie_finding(key, &b_map, b, opts.epsilon) {
                Some(f) => findings.push(f),
                None => structural.push(StructuralDivergence {
                    name: b.name.clone(),
                    kind: "only_in_baseline".into(),
                    detail: "tensor present in baseline but not subject".into(),
                }),
            },
            (None, None) => {}
        }
    }

    for sd in &structural {
        findings.push(
            Finding::new(
                "STRUCTURAL_DIVERGENCE",
                Severity::High,
                format!("{}: {}", sd.kind, sd.name),
            )
            .with_evidence(vec![sd.detail.clone()]),
        );
    }

    // ---- per-layer drift + concentration detection ----
    let (layer_drift, drift_findings) = build_layer_drift(&tensor_drift, opts.mad_k);
    findings.extend(drift_findings);

    // ---- summary + IDENTICAL ----
    let matched = tensor_drift.len() as u64;
    let identical = tensor_drift.iter().filter(|t| t.rel_l2 < 1e-6).count() as u64;
    let identical_fraction = if matched > 0 {
        identical as f64 / matched as f64
    } else {
        0.0
    };
    let worst_drift = tensor_drift
        .iter()
        .map(|t| t.rel_l2)
        .fold(0.0f64, f64::max);

    if structural.is_empty() && matched > 0 && worst_drift < 1e-6 {
        findings.push(Finding::new(
            "IDENTICAL",
            Severity::Info,
            "subject and baseline are identical across all matched tensors (drift ~0)",
        ));
    }
    if deferred_pairs > 0 {
        findings.push(Finding::new(
            "DRIFT_DEFERRED_QUANTIZED",
            Severity::Info,
            format!("{deferred_pairs} matched k-quant/IQ tensor pair(s) skipped (dequant not implemented)"),
        ));
    }

    // Stable ordering.
    tensor_drift.sort_by(|a, b| a.name.cmp(&b.name));

    CompareReport {
        subject: subject.display().to_string(),
        baseline: baseline.display().to_string(),
        arch,
        normalization,
        structural_divergences: structural,
        tensor_drift,
        layer_drift,
        findings,
        summary: Summary {
            matched,
            identical_fraction,
            structural: 0, // filled below
            worst_drift,
        },
    }
    .finalize_summary()
}

impl CompareReport {
    fn finalize_summary(mut self) -> CompareReport {
        self.summary.structural = self.structural_divergences.len() as u64;
        self
    }
}

/// Streaming drift of one matched tensor pair.
fn drift(name: &str, si: ValueIter, bi: ValueIter, eps: f64) -> TensorDrift {
    let mut dot = 0.0;
    let mut diff2 = 0.0;
    let mut n = 0u64;
    let mut changed = 0u64;
    let mut mom_s = Moments::new();
    let mut mom_b = Moments::new();
    let mut max_abs_s = 0.0f64;
    let mut max_abs_b = 0.0f64;

    for (s, b) in si.zip(bi) {
        dot += s * b;
        let d = s - b;
        diff2 += d * d;
        if d.abs() > eps {
            changed += 1;
        }
        mom_s.push(s);
        mom_b.push(b);
        max_abs_s = max_abs_s.max(s.abs());
        max_abs_b = max_abs_b.max(b.abs());
        n += 1;
    }

    let ss = mom_s.sum_squares().max(0.0);
    let bb = mom_b.sum_squares().max(0.0);
    let cosine = if ss > 0.0 && bb > 0.0 {
        (dot / (ss.sqrt() * bb.sqrt())).clamp(-1.0, 1.0)
    } else {
        1.0 // both zero -> treat as identical direction
    };
    let rel_l2 = if bb > 0.0 {
        diff2.sqrt() / bb.sqrt()
    } else if diff2 > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let rms_s = (ss / n.max(1) as f64).sqrt();
    let rms_b = (bb / n.max(1) as f64).sqrt();

    TensorDrift {
        name: name.to_string(),
        cosine,
        rel_l2,
        changed_frac: changed as f64 / n.max(1) as f64,
        delta_max_abs: max_abs_s - max_abs_b,
        delta_rms: rms_s - rms_b,
        delta_std: mom_s.std() - mom_b.std(),
        delta_kurtosis: mom_s.excess_kurtosis() - mom_b.excess_kurtosis(),
        diff2,
        bb,
    }
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Bucket tensor drift into layers and flag CONCENTRATED outliers (a localized
/// spike above the surrounding drift level), not total drift.
fn build_layer_drift(
    tensor_drift: &[TensorDrift],
    mad_k: f64,
) -> (Vec<LayerDrift>, Vec<Finding>) {
    use std::collections::BTreeMap;
    // layer -> (sum diff2, sum bb, dominating tensor by diff2)
    let mut agg: BTreeMap<u64, (f64, f64, (f64, String))> = BTreeMap::new();
    for t in tensor_drift {
        let Some(idx) = crate::profile::layer_index(&t.name) else {
            continue;
        };
        let e = agg.entry(idx).or_insert((0.0, 0.0, (0.0, String::new())));
        e.0 += t.diff2;
        e.1 += t.bb;
        if t.diff2 > e.2 .0 {
            e.2 = (t.diff2, t.name.clone());
        }
    }

    let mut points: Vec<(u64, f64, String)> = agg
        .into_iter()
        .map(|(layer, (d2, bb, dom))| {
            let rel = if bb > 0.0 {
                d2.sqrt() / bb.sqrt()
            } else if d2 > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            (layer, rel, dom.1)
        })
        .collect();
    points.sort_by_key(|p| p.0);

    // Robust concentration: median + MAD of per-layer rel_l2, spikes one-sided.
    let vals: Vec<f64> = points.iter().map(|p| p.1.min(f64::MAX)).collect();
    let med = median(&vals);
    let dev: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
    let mut mad = median(&dev);
    if mad <= f64::EPSILON {
        mad = dev.iter().sum::<f64>() / dev.len().max(1) as f64;
    }

    let mut layer_drift = Vec::new();
    let mut findings = Vec::new();
    for (i, (layer, rel, dom)) in points.iter().enumerate() {
        let mads = if mad > f64::EPSILON {
            (vals[i] - med) / mad
        } else {
            0.0
        };
        let mut anomaly = None;
        if mads > mad_k && *rel > med {
            // Severity from the magnitude of the change (how big, not just how
            // concentrated): a layer that moved by >= its own L2 is drastic.
            let severity = if *rel >= 1.0 {
                Severity::High
            } else if *rel >= 0.3 {
                Severity::Medium
            } else {
                Severity::Low
            };
            findings.push(
                Finding::new(
                    "LAYER_DRIFT_OUTLIER",
                    severity,
                    format!(
                        "layer {layer} drift is a concentrated outlier (rel_l2={rel:.3}, {mads:.1} MADs above the cross-layer drift level) — worth a human look, not a verdict"
                    ),
                )
                .with_evidence(vec![format!("dominant tensor: {dom}")]),
            );
            if !dom.is_empty() {
                findings.push(Finding::new(
                    "TENSOR_DRIFT",
                    severity,
                    format!("tensor '{dom}' dominates the drift of layer {layer}"),
                ));
            }
            anomaly = Some(DriftAnomaly {
                mads,
                severity: severity.as_str().to_string(),
            });
        }
        layer_drift.push(LayerDrift {
            layer: *layer,
            rel_l2: *rel,
            anomaly,
        });
    }

    (layer_drift, findings)
}

fn detect(path: &Path, data: &[u8]) -> Format {
    let head = &data[..data.len().min(16)];
    format::detect(path, head)
}

/// Decide whether a one-sided tensor is explained by weight tying (info) rather
/// than a genuine structural divergence (high).
fn tie_finding(missing_key: &str, side_map: &BTreeMap<String, T>, missing_t: &T, eps: f64) -> Option<Finding> {
    let cands = tied_counterparts(missing_key)?;
    for c in cands {
        let Some(ct) = side_map.get(*c) else {
            continue;
        };
        // Confirm the tie by value-equality on the side that has both, when we
        // can decode them; otherwise accept the serialization convention.
        let tied = if missing_t.shape != ct.shape {
            false
        } else {
            match (missing_t.iter(), ct.iter()) {
                (Some(a), Some(b)) => drift("", a, b, eps).rel_l2 < 1e-6,
                _ => true,
            }
        };
        if tied {
            return Some(
                Finding::new(
                    "TIED_WEIGHT",
                    Severity::Info,
                    format!(
                        "'{}' is tied to '{}' (weight tying) — a serialization convention, not a divergence",
                        missing_t.name, ct.name
                    ),
                )
                .with_evidence(vec!["counterpart present on same side with equal values".into()]),
            );
        }
        return None; // counterpart exists but values differ -> genuine difference
    }
    None
}

fn error_report(subject: &Path, baseline: &Path, msg: &str) -> CompareReport {
    CompareReport {
        subject: subject.display().to_string(),
        baseline: baseline.display().to_string(),
        arch: ArchInfo {
            subject: "?".into(),
            baseline: "?".into(),
            matched: false,
        },
        normalization: Vec::new(),
        structural_divergences: Vec::new(),
        tensor_drift: Vec::new(),
        layer_drift: Vec::new(),
        findings: vec![Finding::new("COMPARE_ERROR", Severity::High, msg.to_string())],
        summary: Summary {
            matched: 0,
            identical_fraction: 0.0,
            structural: 0,
            worst_drift: 0.0,
        },
    }
}

/// Human-readable render.
pub fn render_human(r: &CompareReport, styler: &crate::style::Styler) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} {} {} {}\n",
        styler.bold("compare"),
        styler.dim(&r.subject),
        styler.dim("vs"),
        styler.dim(&r.baseline),
    ));
    out.push_str(&format!(
        "  arch: {} vs {} ({})\n",
        r.arch.subject,
        r.arch.baseline,
        if r.arch.matched {
            styler.green("match")
        } else {
            styler.red("MISMATCH")
        }
    ));
    for note in &r.normalization {
        out.push_str(&format!("  {} {}\n", styler.dim("normalized:"), styler.dim(note)));
    }
    out.push_str(&format!(
        "  {} matched, {} structural divergence(s), worst rel_l2: {:.4}\n",
        r.summary.matched, r.summary.structural, r.summary.worst_drift
    ));

    // Drift sparkline.
    if !r.layer_drift.is_empty() {
        let values: Vec<f64> = r.layer_drift.iter().map(|p| p.rel_l2.min(f64::MAX)).collect();
        let anomalous: Vec<bool> = r.layer_drift.iter().map(|p| p.anomaly.is_some()).collect();
        let labels: Vec<String> = r
            .layer_drift
            .iter()
            .filter(|p| p.anomaly.is_some())
            .map(|p| p.layer.to_string())
            .collect();
        out.push_str("  ");
        out.push_str(
            &crate::profile::render::sparkline_values(
                "drift profile",
                &values,
                &anomalous,
                &labels,
                "rel_l2",
                styler,
            )
            .replace('\n', "\n  "),
        );
        out.push('\n');
    }

    for f in &r.findings {
        out.push_str(&format!(
            "  [{}] {}: {}\n",
            styler.severity(f.severity),
            styler.bold(&f.id),
            f.detail
        ));
        for e in &f.evidence {
            out.push_str(&format!("      {} {e}\n", styler.dim("-")));
        }
    }
    out
}

/// SVG of the drift-per-layer profile.
pub fn render_svg(r: &CompareReport) -> String {
    let values: Vec<f64> = r.layer_drift.iter().map(|p| p.rel_l2.min(1e9)).collect();
    let anomalous: Vec<(usize, f64)> = r
        .layer_drift
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.anomaly.as_ref().map(|a| (i, a.mads)))
        .collect();
    crate::profile::render::svg_values(
        &format!("assay drift profile — rel_l2 ({} layers)", r.layer_drift.len()),
        &values,
        &anomalous,
    )
}
