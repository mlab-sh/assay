//! `assay` — offline-first scanner for ML model artifacts.
//!
//! Entry point: parse args, run the scan (streaming progress to stderr), render
//! the report (colorized on stdout), optionally render the Phase 2 layer
//! profile, and map the aggregate outcome to a CI-friendly exit code.

mod cli;
mod compare;
mod dequant;
mod fingerprint;
mod format;
mod formats;
mod hash;
mod mapio;
mod numeric;
mod phase2;
mod profile;
mod progress;
mod report;
mod scan;
mod secrets;
mod signature;
mod stats;
mod style;
mod values;

use std::io::IsTerminal;
use std::process::ExitCode;

use clap::Parser;

use cli::{Cli, ColorWhen, CommonOpts, Command};
use compare::CompareOpts;
use phase2::Phase2Opts;
use progress::Progress;
use style::Styler;

fn main() -> ExitCode {
    let cli = Cli::parse();

    // `compare` is its own pipeline.
    if let Command::Compare {
        subject,
        baseline,
        json,
        svg,
        mad_k,
        epsilon,
        force,
        fail_on,
        color,
    } = cli.command
    {
        return run_compare(subject, baseline, json, svg, mad_k, epsilon, force, fail_on, color);
    }

    let (path, bundle, key, common) = match cli.command {
        Command::Scan { path, common } => (path, None, None, common),
        Command::Verify {
            path,
            bundle,
            key,
            common,
        } => (path, bundle, key, common),
        Command::Compare { .. } => unreachable!("handled above"),
    };

    let phase2_opts = common.deep_enabled().then_some(Phase2Opts {
        mad_k: common.mad_k,
        scan_tensor_entropy: common.scan_tensor_entropy,
    });

    let mut progress = make_progress(&common);
    let report = scan::run(
        &path,
        bundle.as_deref(),
        key.as_deref(),
        phase2_opts.as_ref(),
        &mut progress,
    );

    let stdout_color = color_for(common.color, std::io::stdout().is_terminal());
    let styler = Styler::new(stdout_color);

    if common.json {
        println!("{}", report.to_json());
    } else {
        print!("{}", report.to_human(&styler));
        if common.profile {
            render_profiles(&report, &styler);
        }
    }

    if let Some(svg_path) = &common.svg {
        write_svg(&report, svg_path);
    }

    // exit_code() returns the worst outcome (3 > 2 > 1 > 0), per the README table.
    let code = report.exit_code(common.fail_on);
    ExitCode::from(code as u8)
}

#[allow(clippy::too_many_arguments)]
fn run_compare(
    subject: std::path::PathBuf,
    baseline: std::path::PathBuf,
    json: bool,
    svg: Option<std::path::PathBuf>,
    mad_k: f64,
    epsilon: f64,
    force: bool,
    fail_on: report::Severity,
    color: ColorWhen,
) -> ExitCode {
    let report = compare::run(
        &subject,
        &baseline,
        &CompareOpts {
            mad_k,
            epsilon,
            force,
        },
    );

    if json {
        println!("{}", report.to_json());
    } else {
        let styler = Styler::new(color_for(color, std::io::stdout().is_terminal()));
        print!("{}", compare::render_human(&report, &styler));
    }

    if let Some(path) = &svg {
        match std::fs::write(path, compare::render_svg(&report)) {
            Ok(()) => eprintln!("drift-profile SVG written to {}", path.display()),
            Err(e) => eprintln!("could not write SVG to {}: {e}", path.display()),
        }
    }

    ExitCode::from(report.exit_code(fail_on) as u8)
}

/// Print a per-artifact layer-profile sparkline to stdout.
fn render_profiles(report: &report::ScanReport, styler: &Styler) {
    for a in &report.artifacts {
        if let Some(points) = &a.layer_profile {
            if !points.is_empty() {
                println!("\n{}", styler.dim(&a.artifact));
                println!("{}", profile::render::sparkline(points, "l2", styler));
            }
        }
    }
}

/// Write the layer profile of the first artifact that has one to an SVG file.
fn write_svg(report: &report::ScanReport, path: &std::path::Path) {
    for a in &report.artifacts {
        if let Some(points) = &a.layer_profile {
            if !points.is_empty() {
                let svg = profile::render::svg(points, "l2");
                match std::fs::write(path, svg) {
                    Ok(()) => eprintln!("layer-profile SVG written to {}", path.display()),
                    Err(e) => eprintln!("could not write SVG to {}: {e}", path.display()),
                }
                return;
            }
        }
    }
    eprintln!("no layer profile available to render as SVG");
}

/// Build the stderr progress reporter, honoring `--no-progress`, `--color`,
/// `NO_COLOR`, and whether stderr is a terminal.
fn make_progress(common: &CommonOpts) -> Progress {
    let stderr_tty = std::io::stderr().is_terminal();
    let enabled = !common.no_progress && stderr_tty;
    let styler = Styler::new(color_for(common.color, stderr_tty));
    Progress::new(enabled, styler)
}

fn color_for(when: ColorWhen, is_tty: bool) -> bool {
    match when {
        ColorWhen::Always => true,
        ColorWhen::Never => false,
        ColorWhen::Auto => is_tty && std::env::var_os("NO_COLOR").is_none(),
    }
}
