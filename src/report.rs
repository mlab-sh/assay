//! Report data model + rendering (human + JSON).
//!
//! The `serde` field names here are chosen to match the sample JSON in the
//! project README exactly, so the JSON output is a stable, documented contract.

use std::collections::BTreeMap;

use serde::Serialize;

/// Severity of a finding. Ordering matters: `--fail-on` compares against it,
/// and the aggregate exit code keys off the worst severity present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
            Severity::Critical => "critical",
        }
    }
}

impl std::str::FromStr for Severity {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "info" => Ok(Severity::Info),
            "low" => Ok(Severity::Low),
            "medium" => Ok(Severity::Medium),
            "high" => Ok(Severity::High),
            "critical" => Ok(Severity::Critical),
            other => Err(format!(
                "unknown severity '{other}' (expected info|low|medium|high|critical)"
            )),
        }
    }
}

/// A single thing `assay` noticed about an artifact.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub id: String,
    pub severity: Severity,
    pub detail: String,
    /// Concrete supporting evidence (opcode traces, offsets, …). Omitted from
    /// JSON when empty, matching the README sample.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub evidence: Vec<String>,
}

impl Finding {
    pub fn new(id: &str, severity: Severity, detail: impl Into<String>) -> Self {
        Finding {
            id: id.to_string(),
            severity,
            detail: detail.into(),
            evidence: Vec::new(),
        }
    }

    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }
}

/// Overall trust call for one artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// No findings at/above an actionable severity.
    Clean,
    /// Can run code / carries unreviewed injection surface; handle with care.
    Untrusted,
    /// Could not be parsed as the format it claims to be.
    Malformed,
    /// Internal error while processing (IO, etc.).
    Error,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Clean => "clean",
            Verdict::Untrusted => "untrusted",
            Verdict::Malformed => "malformed",
            Verdict::Error => "error",
        }
    }
}

/// Hashes computed for an artifact.
///
/// The two top-level digests answer different questions and neither replaces
/// the other:
///
/// * `manifest` is the rename/repack-stable *model identity*: it covers tensor
///   names, dtypes, shapes and content, and nothing else. Two files with the
///   same manifest hash hold the same model, but they are not necessarily the
///   same bytes.
/// * `file` is the digest of the artifact exactly as it sits on disk, every
///   byte included. It is what you pin when you want to know that nothing at
///   all changed, including bytes no tensor claims.
///
/// `per_tensor` is included only when non-empty.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Hashes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub per_tensor: BTreeMap<String, String>,
}

impl Hashes {
    pub fn is_empty(&self) -> bool {
        self.manifest.is_none() && self.file.is_none() && self.per_tensor.is_empty()
    }
}

/// Phase 2 statistics block (present only with `--deep`).
#[derive(Debug, Clone, Serialize)]
pub struct StatsBlock {
    pub per_tensor: Vec<crate::stats::PerTensorStats>,
}

/// Report for a single artifact.
#[derive(Debug, Clone, Serialize)]
pub struct ArtifactReport {
    pub artifact: String,
    pub format: String,
    pub verdict: Verdict,
    pub findings: Vec<Finding>,
    #[serde(skip_serializing_if = "Hashes::is_empty")]
    pub hashes: Hashes,
    pub signature: String,

    // --- Phase 2 (additive; absent unless `--deep`) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<StatsBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_profile: Option<Vec<crate::profile::ProfilePoint>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<crate::fingerprint::Fingerprint>,
}

impl ArtifactReport {
    pub fn new(artifact: impl Into<String>, format: &str) -> Self {
        ArtifactReport {
            artifact: artifact.into(),
            format: format.to_string(),
            verdict: Verdict::Clean,
            findings: Vec::new(),
            hashes: Hashes::default(),
            signature: "unsigned".to_string(),
            stats: None,
            layer_profile: None,
            fingerprint: None,
        }
    }

    pub fn push(&mut self, f: Finding) {
        self.findings.push(f);
    }

    /// Highest severity among findings, if any.
    pub fn max_severity(&self) -> Option<Severity> {
        self.findings.iter().map(|f| f.severity).max()
    }
}

/// Aggregate report for a whole scan (one or many artifacts).
#[derive(Debug, Clone, Default, Serialize)]
pub struct ScanReport {
    pub artifacts: Vec<ArtifactReport>,
}

impl ScanReport {
    pub fn any_malformed(&self) -> bool {
        self.artifacts
            .iter()
            .any(|a| a.verdict == Verdict::Malformed)
    }

    pub fn any_error(&self) -> bool {
        self.artifacts.iter().any(|a| a.verdict == Verdict::Error)
    }

    pub fn max_severity(&self) -> Option<Severity> {
        self.artifacts.iter().filter_map(|a| a.max_severity()).max()
    }

    /// Worst-outcome-wins exit code, per the README table.
    /// Precedence: 3 (internal) > 2 (malformed) > 1 (findings) > 0 (clean).
    pub fn exit_code(&self, fail_on: Severity) -> i32 {
        if self.any_error() {
            return 3;
        }
        if self.any_malformed() {
            return 2;
        }
        match self.max_severity() {
            Some(sev) if sev >= fail_on => 1,
            _ => 0,
        }
    }

    /// Render as pretty JSON. A single-artifact scan emits the bare artifact
    /// object (matching the README sample); multi-artifact emits the wrapper.
    pub fn to_json(&self) -> String {
        if self.artifacts.len() == 1 {
            serde_json::to_string_pretty(&self.artifacts[0])
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        } else {
            serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
        }
    }

    /// Render a concise, human-readable report, colorized via `styler`.
    pub fn to_human(&self, styler: &crate::style::Styler) -> String {
        let mut out = String::new();
        for a in &self.artifacts {
            out.push_str(&format!(
                "{}  {}  {} {}\n",
                styler.bold(&a.artifact),
                styler.dim(&format!("[{}]", a.format)),
                styler.dim("->"),
                styler.verdict(a.verdict),
            ));
            if let Some(m) = &a.hashes.manifest {
                out.push_str(&format!(
                    "  {} {}\n",
                    styler.dim("manifest:"),
                    styler.dim(m)
                ));
            }
            if let Some(f) = &a.hashes.file {
                out.push_str(&format!("  {} {}\n", styler.dim("file:"), styler.dim(f)));
            }
            out.push_str(&format!("  {} {}\n", styler.dim("signature:"), a.signature));
            for f in &a.findings {
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
            out.push('\n');
        }
        let summary = match self.max_severity() {
            _ if self.any_error() => styler.red("internal error"),
            _ if self.any_malformed() => styler.yellow("malformed artifact(s) present"),
            Some(sev) => format!("worst finding: {}", styler.severity(sev)),
            None => styler.green("clean"),
        };
        out.push_str(&format!(
            "scanned {} artifact(s); {}\n",
            self.artifacts.len(),
            summary
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(verdict: Verdict, findings: &[(&str, Severity)]) -> ArtifactReport {
        let mut a = ArtifactReport::new("m.safetensors", "safetensors");
        a.verdict = verdict;
        for (id, sev) in findings {
            a.push(Finding::new(id, *sev, "detail"));
        }
        a
    }

    fn scan(artifacts: Vec<ArtifactReport>) -> ScanReport {
        ScanReport { artifacts }
    }

    #[test]
    fn severity_orders_from_info_to_critical() {
        assert!(Severity::Info < Severity::Low);
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn severity_parses_case_and_space_insensitively() {
        assert_eq!("  HIGH ".parse::<Severity>().unwrap(), Severity::High);
        assert_eq!("info".parse::<Severity>().unwrap(), Severity::Info);
        assert!("bogus".parse::<Severity>().is_err());
    }

    #[test]
    fn exit_code_respects_the_fail_on_threshold() {
        let r = scan(vec![artifact(Verdict::Clean, &[("X", Severity::Medium)])]);
        assert_eq!(r.exit_code(Severity::High), 0);
        assert_eq!(r.exit_code(Severity::Medium), 1);
        assert_eq!(r.exit_code(Severity::Low), 1);
    }

    #[test]
    fn exit_code_precedence_is_error_then_malformed_then_findings() {
        let clean = scan(vec![artifact(Verdict::Clean, &[("I", Severity::Info)])]);
        assert_eq!(clean.exit_code(Severity::High), 0);

        let flagged = scan(vec![artifact(Verdict::Untrusted, &[("H", Severity::High)])]);
        assert_eq!(flagged.exit_code(Severity::High), 1);

        // A malformed artifact outranks a high finding elsewhere in the scan.
        let malformed = scan(vec![
            artifact(Verdict::Untrusted, &[("H", Severity::High)]),
            artifact(Verdict::Malformed, &[("M", Severity::Medium)]),
        ]);
        assert_eq!(malformed.exit_code(Severity::High), 2);

        // An internal error outranks everything.
        let errored = scan(vec![
            artifact(Verdict::Malformed, &[("M", Severity::Medium)]),
            artifact(Verdict::Error, &[("IO_ERROR", Severity::High)]),
        ]);
        assert_eq!(errored.exit_code(Severity::High), 3);
    }

    #[test]
    fn max_severity_is_the_worst_across_all_artifacts() {
        let r = scan(vec![
            artifact(Verdict::Clean, &[("A", Severity::Low)]),
            artifact(
                Verdict::Untrusted,
                &[("B", Severity::High), ("C", Severity::Info)],
            ),
        ]);
        assert_eq!(r.max_severity(), Some(Severity::High));
        assert_eq!(
            scan(vec![artifact(Verdict::Clean, &[])]).max_severity(),
            None
        );
    }

    #[test]
    fn single_artifact_json_is_the_bare_object() {
        let r = scan(vec![artifact(
            Verdict::Untrusted,
            &[("PICKLE_RCE_RISK", Severity::High)],
        )]);
        let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(v["artifact"], "m.safetensors");
        assert_eq!(v["verdict"], "untrusted");
        assert_eq!(v["findings"][0]["id"], "PICKLE_RCE_RISK");
        assert_eq!(v["findings"][0]["severity"], "high");
        assert_eq!(v["signature"], "unsigned");
        assert!(v.get("artifacts").is_none());
    }

    #[test]
    fn multi_artifact_json_is_wrapped() {
        let r = scan(vec![
            artifact(Verdict::Clean, &[]),
            artifact(Verdict::Untrusted, &[("X", Severity::High)]),
        ]);
        let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(v["artifacts"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn empty_optional_blocks_are_omitted_from_json() {
        let r = scan(vec![artifact(Verdict::Clean, &[("X", Severity::Info)])]);
        let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        // No evidence, no hashes, and none of the weight-analysis blocks.
        assert!(v["findings"][0].get("evidence").is_none());
        assert!(v.get("hashes").is_none());
        assert!(v.get("stats").is_none());
        assert!(v.get("layer_profile").is_none());
        assert!(v.get("fingerprint").is_none());
    }

    #[test]
    fn both_digests_are_serialized_and_named() {
        let mut a = ArtifactReport::new("m.safetensors", "safetensors");
        a.hashes.manifest = Some("blake3:aaa".into());
        a.hashes.file = Some("blake3:bbb".into());
        let v: serde_json::Value = serde_json::from_str(&scan(vec![a]).to_json()).unwrap();
        assert_eq!(v["hashes"]["manifest"], "blake3:aaa");
        assert_eq!(v["hashes"]["file"], "blake3:bbb");
    }

    #[test]
    fn a_file_digest_alone_is_enough_to_emit_the_hashes_block() {
        // Pickle artifacts have no tensors, but they are still pinnable.
        let mut a = ArtifactReport::new("m.bin", "pickle");
        assert!(a.hashes.is_empty());
        a.hashes.file = Some("blake3:ccc".into());
        assert!(!a.hashes.is_empty());
        let v: serde_json::Value = serde_json::from_str(&scan(vec![a]).to_json()).unwrap();
        assert_eq!(v["hashes"]["file"], "blake3:ccc");
        assert!(v["hashes"].get("manifest").is_none());
    }

    #[test]
    fn the_human_report_prints_both_digests() {
        let styler = crate::style::Styler::new(false);
        let mut a = ArtifactReport::new("m.safetensors", "safetensors");
        a.hashes.manifest = Some("blake3:aaa".into());
        a.hashes.file = Some("blake3:bbb".into());
        let out = scan(vec![a]).to_human(&styler);
        assert!(out.contains("manifest: blake3:aaa"), "{out}");
        assert!(out.contains("file: blake3:bbb"), "{out}");
    }

    #[test]
    fn evidence_is_serialized_when_present() {
        let mut a = ArtifactReport::new("m.bin", "pickle");
        a.push(Finding::new("X", Severity::High, "d").with_evidence(vec!["opcode REDUCE".into()]));
        let v: serde_json::Value = serde_json::from_str(&scan(vec![a]).to_json()).unwrap();
        assert_eq!(v["findings"][0]["evidence"][0], "opcode REDUCE");
    }

    #[test]
    fn human_report_lists_every_finding_and_a_summary() {
        let styler = crate::style::Styler::new(false);
        let r = scan(vec![artifact(
            Verdict::Untrusted,
            &[("A", Severity::High), ("B", Severity::Info)],
        )]);
        let out = r.to_human(&styler);
        assert!(out.contains("m.safetensors"));
        assert!(out.contains("[safetensors]"));
        assert!(out.contains("[high] A: detail"));
        assert!(out.contains("[info] B: detail"));
        assert!(out.contains("scanned 1 artifact(s); worst finding: high"));
    }

    #[test]
    fn human_report_says_clean_when_there_is_nothing_to_report() {
        let styler = crate::style::Styler::new(false);
        let out = scan(vec![artifact(Verdict::Clean, &[])]).to_human(&styler);
        assert!(out.contains("scanned 1 artifact(s); clean"), "{out}");
    }
}
