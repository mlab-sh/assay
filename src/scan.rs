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

    let has_safetensors = artifacts
        .iter()
        .any(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"));

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

        // Whole-file digest, for every format including pickle and unknown.
        // The manifest hash deliberately covers only tensor identity and
        // content, so it cannot distinguish two files that differ in bytes no
        // tensor claims. This one can.
        ar.hashes.file = Some(crate::hash::tagged(&crate::hash::blake3_hex(data)));

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

    // Repo level: weights cannot execute, but the code shipped next to them
    // can, and a loader will run it before it reads a single tensor.
    for ar in crate::remote_code::scan(path) {
        report.artifacts.push(ar);
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Styler;

    /// A unique temp dir per test, cleaned on entry so reruns are hermetic.
    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("assay-scan-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// Minimal one-tensor safetensors file.
    fn safetensors_bytes() -> Vec<u8> {
        let header = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 8]);
        out
    }

    /// `os.system("echo hi")` pickle: the canonical code-execution payload.
    fn rce_pickle() -> Vec<u8> {
        let mut p = vec![0x80, 0x02];
        p.extend_from_slice(b"cos\nsystem\n");
        p.push(b'U');
        let arg = b"echo hi";
        p.push(arg.len() as u8);
        p.extend_from_slice(arg);
        p.push(b'\x85');
        p.push(b'R');
        p.push(b'.');
        p
    }

    fn silent_progress() -> Progress {
        Progress::new(false, Styler::new(false))
    }

    fn run_scan(path: &Path) -> ScanReport {
        run(path, None, None, None, &mut silent_progress())
    }

    fn ids(ar: &ArtifactReport) -> Vec<&str> {
        ar.findings.iter().map(|f| f.id.as_str()).collect()
    }

    #[test]
    fn a_file_path_resolves_to_exactly_that_file() {
        let dir = tmpdir("single");
        let p = write(&dir, "model.safetensors", &safetensors_bytes());
        write(&dir, "other.safetensors", &safetensors_bytes());
        assert_eq!(collect_artifacts(&p), vec![p]);
    }

    #[test]
    fn directory_walk_is_recursive_sorted_and_filtered() {
        let dir = tmpdir("walk");
        write(&dir, "model.safetensors", &safetensors_bytes());
        write(&dir, "nested/deep.gguf", b"GGUF");
        write(&dir, "pytorch_model.bin", &rce_pickle());
        // Non-artifacts that live in every real model repo.
        write(&dir, "README.md", b"# hi");
        write(&dir, "config.json", b"{}");
        write(&dir, "tokenizer.json", b"{}");

        let found = collect_artifacts(&dir);
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 3, "got {names:?}");
        assert!(names.contains(&"deep.gguf".to_string()));
        assert!(names.contains(&"model.safetensors".to_string()));
        assert!(names.contains(&"pytorch_model.bin".to_string()));
        // Sorted, so report order is deterministic across filesystems.
        let mut sorted = found.clone();
        sorted.sort();
        assert_eq!(found, sorted);
    }

    #[test]
    fn a_path_with_nothing_scannable_reports_no_artifacts() {
        let dir = tmpdir("empty");
        write(&dir, "README.md", b"# hi");
        let report = run_scan(&dir);
        assert_eq!(report.artifacts.len(), 1);
        assert_eq!(report.artifacts[0].verdict, Verdict::Error);
        assert_eq!(ids(&report.artifacts[0]), vec!["NO_ARTIFACTS"]);
    }

    #[test]
    fn a_pickle_next_to_a_safetensors_gets_the_safe_alternative_hint() {
        let dir = tmpdir("alt");
        write(&dir, "model.safetensors", &safetensors_bytes());
        write(&dir, "pytorch_model.bin", &rce_pickle());

        let report = run_scan(&dir);
        let pickle = report
            .artifacts
            .iter()
            .find(|a| a.format == "pickle")
            .expect("pickle artifact");
        assert_eq!(pickle.verdict, Verdict::Untrusted);
        assert!(ids(pickle).contains(&"PICKLE_RCE_RISK"));
        assert!(ids(pickle).contains(&"SAFE_ALTERNATIVE_AVAILABLE"));

        let st = report
            .artifacts
            .iter()
            .find(|a| a.format == "safetensors")
            .expect("safetensors artifact");
        assert_eq!(st.verdict, Verdict::Clean);
        assert!(!ids(st).contains(&"SAFE_ALTERNATIVE_AVAILABLE"));
    }

    #[test]
    fn a_lone_pickle_gets_no_alternative_hint() {
        let dir = tmpdir("lonepickle");
        write(&dir, "pytorch_model.bin", &rce_pickle());
        let report = run_scan(&dir);
        assert!(!ids(&report.artifacts[0]).contains(&"SAFE_ALTERNATIVE_AVAILABLE"));
    }

    #[test]
    fn an_unidentifiable_artifact_is_malformed_not_guessed() {
        let dir = tmpdir("unknown");
        // `.ckpt` gets scanned, but the bytes are not a pickle stream.
        let p = write(&dir, "weights.ckpt", b"not a model at all");
        let report = run_scan(&p);
        assert_eq!(report.artifacts.len(), 1);
        // The extension routes it to the pickle parser, which refuses it.
        assert_ne!(report.artifacts[0].verdict, Verdict::Clean);
    }

    #[test]
    fn scanning_without_deep_leaves_the_weight_blocks_empty() {
        let dir = tmpdir("nodeep");
        let p = write(&dir, "model.safetensors", &safetensors_bytes());
        let report = run_scan(&p);
        let a = &report.artifacts[0];
        assert!(a.stats.is_none());
        assert!(a.layer_profile.is_none());
        assert!(a.fingerprint.is_none());
        // The container answer is still complete: hashed and signature-checked.
        assert!(a.hashes.manifest.is_some());
        assert_eq!(a.signature, "unsigned");
    }

    #[test]
    fn every_artifact_gets_a_whole_file_digest() {
        let dir = tmpdir("filehash");
        let st = write(&dir, "model.safetensors", &safetensors_bytes());
        let pk = write(&dir, "pytorch_model.bin", &rce_pickle());
        let unknown = write(&dir, "weights.ckpt", b"not a model at all");

        for p in [&st, &pk, &unknown] {
            let report = run_scan(p);
            let h = &report.artifacts[0].hashes.file;
            assert!(
                h.as_deref().is_some_and(|h| h.starts_with("blake3:")),
                "{} has no file digest",
                p.display()
            );
        }
    }

    /// The point of the whole-file digest: the manifest hash covers the model,
    /// not the file. Bytes appended after the last tensor leave the manifest
    /// untouched, so only the file digest can tell the two apart.
    #[test]
    fn appended_bytes_keep_the_manifest_but_change_the_file_digest() {
        let dir = tmpdir("twin");
        let clean = write(&dir, "clean.safetensors", &safetensors_bytes());
        let mut tampered_bytes = safetensors_bytes();
        tampered_bytes.extend_from_slice(b"PK\x03\x04appended archive");
        let tampered = write(&dir, "polyglot.safetensors", &tampered_bytes);

        let a = run_scan(&clean);
        let b = run_scan(&tampered);
        let (ha, hb) = (&a.artifacts[0].hashes, &b.artifacts[0].hashes);

        assert_eq!(
            ha.manifest, hb.manifest,
            "same tensors, same model identity"
        );
        assert_ne!(ha.file, hb.file, "different bytes must hash differently");
        assert!(ids(&b.artifacts[0]).contains(&"ST_UNREFERENCED_BYTES"));
    }

    #[test]
    fn the_file_digest_covers_content_not_the_filename() {
        let dir = tmpdir("filerename");
        let a = write(&dir, "model.safetensors", &safetensors_bytes());
        let b = write(&dir, "renamed.safetensors", &safetensors_bytes());
        assert_eq!(
            run_scan(&a).artifacts[0].hashes.file,
            run_scan(&b).artifacts[0].hashes.file
        );
    }

    #[test]
    fn deep_scan_attaches_the_weight_blocks() {
        let dir = tmpdir("deep");
        let p = write(&dir, "model.safetensors", &safetensors_bytes());
        let opts = Phase2Opts {
            mad_k: 5.0,
            scan_tensor_entropy: false,
        };
        let report = run(&p, None, None, Some(&opts), &mut silent_progress());
        assert!(report.artifacts[0].stats.is_some());
    }
}
