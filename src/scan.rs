//! Scan orchestration: resolve a path to a set of artifacts, dispatch each to
//! the right format analyzer, attach signature status + (optionally) Phase 2
//! weight analysis, and surface cross-artifact signals.
//!
//! Artifacts are memory-mapped, not read into a heap buffer, so peak RAM stays
//! well under model size even for very large models. Both phases operate on the
//! same `&[u8]` view.

use std::path::{Path, PathBuf};

use crate::format::{self, Format};
use crate::formats::{gguf, pickle, safetensors};
use crate::mapio::map_file;
use crate::phase2::{self, Phase2Opts};
use crate::progress::Progress;
use crate::report::{ArtifactReport, Finding, ScanReport, Severity, StatsBlock, Verdict};
use crate::signature;

/// How much of a file to read for format sniffing.
const SNIFF_BYTES: usize = 16;

/// Scan a file or directory. `bundle`/`key` come from the `verify` command and
/// apply to every artifact. `phase2` enables Phase 2 weight analysis.
pub fn run(
    path: &Path,
    bundle: Option<&Path>,
    key: Option<&Path>,
    phase2_opts: Option<&Phase2Opts>,
    progress: &mut Progress,
) -> ScanReport {
    let mut report = ScanReport::default();
    let artifacts = collect_artifacts(path);
    let total = artifacts.len();

    let has_safetensors = artifacts.iter().any(|p| {
        p.extension().and_then(|e| e.to_str()) == Some("safetensors")
    });

    for (i, artifact) in artifacts.iter().enumerate() {
        let idx = i + 1;
        let name = artifact.display().to_string();
        let size = std::fs::metadata(artifact).map(|m| m.len()).unwrap_or(0);
        progress.file_started(idx, total, &name, size);

        let bytes = match map_file(artifact) {
            Ok(b) => b,
            Err(e) => {
                let mut ar = ArtifactReport::new(name.clone(), "unknown");
                ar.verdict = Verdict::Error;
                ar.push(Finding::new(
                    "IO_ERROR",
                    Severity::High,
                    format!("could not read artifact: {e}"),
                ));
                progress.file_finished(idx, total, &name, size, ar.verdict, ar.findings.len());
                report.artifacts.push(ar);
                continue;
            }
        };
        let data: &[u8] = &bytes;
        let head_len = data.len().min(SNIFF_BYTES);
        let fmt = format::detect(artifact, &data[..head_len]);

        // --- Phase 1 ---
        let mut ar = match fmt {
            Format::Pickle => pickle::analyze(&name, data),
            Format::Safetensors => safetensors::analyze(&name, data),
            Format::Gguf => gguf::analyze(&name, data),
            Format::Unknown => {
                let mut ar = ArtifactReport::new(name.clone(), "unknown");
                ar.verdict = Verdict::Malformed;
                ar.push(Finding::new(
                    "UNKNOWN_FORMAT",
                    Severity::Medium,
                    "could not identify this artifact as safetensors, GGUF, or pickle",
                ));
                ar
            }
        };

        // Cross-artifact signal: prefer the safe alternative over a pickle.
        if ar.format == "pickle" && has_safetensors {
            ar.push(Finding::new(
                "SAFE_ALTERNATIVE_AVAILABLE",
                Severity::Info,
                "a safetensors artifact is present in the same repo; prefer it",
            ));
        }

        // Signature evaluation.
        let computed = ar.hashes.manifest.clone();
        let outcome = signature::evaluate(artifact, computed.as_deref(), bundle, key);
        ar.signature = outcome.status;
        for f in outcome.findings {
            ar.push(f);
        }

        // --- Phase 2 (additive; never changes the Phase 1 verdict) ---
        if let Some(opts) = phase2_opts {
            if matches!(fmt, Format::Safetensors | Format::Gguf) {
                let p2 = phase2::run(ar.format.as_str(), data, artifact, opts);
                for f in p2.findings {
                    ar.push(f);
                }
                if !p2.per_tensor.is_empty() {
                    ar.stats = Some(StatsBlock {
                        per_tensor: p2.per_tensor,
                    });
                }
                if !p2.layer_profile.is_empty() {
                    ar.layer_profile = Some(p2.layer_profile);
                }
                ar.fingerprint = p2.fingerprint;
            }
        }

        progress.file_finished(idx, total, &name, size, ar.verdict, ar.findings.len());
        report.artifacts.push(ar);
    }
    progress.finish();

    if report.artifacts.is_empty() {
        let mut ar = ArtifactReport::new(path.display().to_string(), "unknown");
        ar.verdict = Verdict::Error;
        ar.push(Finding::new(
            "NO_ARTIFACTS",
            Severity::Info,
            "no scannable model artifacts found at the given path",
        ));
        report.artifacts.push(ar);
    }

    report
}

/// Resolve a path into the list of artifact files to scan.
fn collect_artifacts(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut out = Vec::new();
    if path.is_dir() {
        walk(path, &mut out);
        out.sort();
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk(&p, out);
        } else if format::is_candidate_artifact(&p) {
            out.push(p);
        }
    }
}
