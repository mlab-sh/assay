//! Command-line interface definitions.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::report::Severity;

#[derive(Parser, Debug)]
#[command(
    name = "assay",
    version,
    about = "Assay the weights before you trust them — scan ML model artifacts for supply-chain safety.",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Scan a model file or directory for provenance & integrity issues.
    Scan {
        /// File or HF-style model directory to scan.
        path: PathBuf,

        #[command(flatten)]
        common: CommonOpts,
    },

    /// Verify a signature / provenance bundle alongside the weights.
    Verify {
        /// File or model directory to verify.
        path: PathBuf,

        /// Explicit signature/provenance bundle to verify against.
        #[arg(long, value_name = "FILE")]
        bundle: Option<PathBuf>,

        /// Public key (ed25519, raw or hex) for detached-signature verification.
        #[arg(long, value_name = "FILE")]
        key: Option<PathBuf>,

        #[command(flatten)]
        common: CommonOpts,
    },

    /// Differential weight analysis: how SUBJECT differs from a known-good BASELINE.
    Compare {
        /// The model under inspection.
        subject: PathBuf,
        /// The known-good reference of the same architecture.
        baseline: PathBuf,

        /// Emit a machine-readable JSON report.
        #[arg(long)]
        json: bool,

        /// Write the drift-per-layer chart as SVG to this path.
        #[arg(long, value_name = "FILE")]
        svg: Option<PathBuf>,

        /// Concentration threshold for layer drift, in MADs.
        #[arg(long, value_name = "K", default_value = "5.0")]
        mad_k: f64,

        /// Elements differing by more than this count toward `changed_frac`.
        #[arg(long, value_name = "EPS", default_value = "1e-6")]
        epsilon: f64,

        /// Compare even across mismatched architectures (output is unreliable).
        #[arg(long)]
        force: bool,

        /// Exit non-zero if any finding is at or above this severity.
        #[arg(long, value_name = "SEVERITY", default_value = "high")]
        fail_on: Severity,

        /// When to colorize output.
        #[arg(long, value_name = "WHEN", default_value = "auto")]
        color: ColorWhen,
    },
}

/// Flags shared by every subcommand.
#[derive(Args, Debug)]
pub struct CommonOpts {
    /// Emit a machine-readable JSON report.
    #[arg(long)]
    pub json: bool,

    /// Exit non-zero if any finding is at or above this severity.
    #[arg(long, value_name = "SEVERITY", default_value = "high")]
    pub fail_on: Severity,

    /// When to colorize output.
    #[arg(long, value_name = "WHEN", default_value = "auto")]
    pub color: ColorWhen,

    /// Disable the real-time progress display on stderr.
    #[arg(long)]
    pub no_progress: bool,

    // --- Phase 2: weight inspection (signals, not verdicts) ---
    /// Enable Phase 2 weight analysis (per-tensor stats, layer profile, …).
    #[arg(long, visible_alias = "stats")]
    pub deep: bool,

    /// Emit the per-layer profile (implies --deep).
    #[arg(long)]
    pub profile: bool,

    /// Write the layer-profile chart as SVG to this path (implies --deep).
    #[arg(long, value_name = "FILE")]
    pub svg: Option<PathBuf>,

    /// Robust anomaly threshold, in MADs from the cross-layer median.
    #[arg(long, value_name = "K", default_value = "5.0")]
    pub mad_k: f64,

    /// [experimental] flag near-maximal-entropy integer tensor regions.
    #[arg(long)]
    pub scan_tensor_entropy: bool,
}

impl CommonOpts {
    /// Phase 2 runs if any of its flags were requested.
    pub fn deep_enabled(&self) -> bool {
        self.deep || self.profile || self.svg.is_some() || self.scan_tensor_entropy
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorWhen {
    Auto,
    Always,
    Never,
}
