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
    /// Can run code / carries unreviewed injection surface — handle with care.
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

/// Hashes computed for an artifact. `manifest` is the rename/repack-stable
/// identity anchor; `per_tensor` is included only when non-empty.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Hashes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub per_tensor: BTreeMap<String, String>,
}

impl Hashes {
    pub fn is_empty(&self) -> bool {
        self.manifest.is_none() && self.per_tensor.is_empty()
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
        self.artifacts.iter().any(|a| a.verdict == Verdict::Malformed)
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
            serde_json::to_string_pretty(self)
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
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
                out.push_str(&format!("  {} {}\n", styler.dim("manifest:"), styler.dim(m)));
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
