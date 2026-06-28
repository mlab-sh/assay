//! Real-time scan progress + stats, emitted to **stderr** so it never
//! contaminates the stdout report (human or JSON). Disabled automatically when
//! stderr is not a TTY, or via `--no-progress`.

use std::io::Write;
use std::time::Instant;

use crate::report::Verdict;
use crate::style::Styler;

pub struct Progress {
    enabled: bool,
    styler: Styler,
    overall_start: Instant,
    file_start: Instant,
    // running tallies
    bytes_total: u64,
    clean: u32,
    untrusted: u32,
    malformed: u32,
    error: u32,
}

impl Progress {
    pub fn new(enabled: bool, styler: Styler) -> Self {
        let now = Instant::now();
        Progress {
            enabled,
            styler,
            overall_start: now,
            file_start: now,
            bytes_total: 0,
            clean: 0,
            untrusted: 0,
            malformed: 0,
            error: 0,
        }
    }

    /// Announce the artifact about to be scanned (line stays open, no newline).
    pub fn file_started(&mut self, idx: usize, total: usize, name: &str, size: u64) {
        self.file_start = Instant::now();
        if !self.enabled {
            return;
        }
        let counter = self.styler.dim(&format!("[{idx}/{total}]"));
        eprint!(
            "\r{counter} scanning {} {} … ",
            name,
            self.styler.dim(&format!("({})", human_size(size)))
        );
        let _ = std::io::stderr().flush();
    }

    /// Close out the current artifact's line with its verdict and timing.
    pub fn file_finished(&mut self, idx: usize, total: usize, name: &str, size: u64, verdict: Verdict, n_findings: usize) {
        self.bytes_total += size;
        match verdict {
            Verdict::Clean => self.clean += 1,
            Verdict::Untrusted => self.untrusted += 1,
            Verdict::Malformed => self.malformed += 1,
            Verdict::Error => self.error += 1,
        }
        if !self.enabled {
            return;
        }
        let elapsed = self.file_start.elapsed();
        let counter = self.styler.dim(&format!("[{idx}/{total}]"));
        let findings = if n_findings == 0 {
            self.styler.dim("no findings")
        } else {
            self.styler.dim(&format!("{n_findings} finding(s)"))
        };
        // \r + clear-to-end-of-line, then the final state for this artifact.
        eprintln!(
            "\r\x1b[K{counter} {} {} {} {}",
            name,
            self.styler.verdict(verdict),
            findings,
            self.styler.dim(&format!("({})", human_duration(elapsed)))
        );
    }

    /// Final aggregate stats line.
    pub fn finish(&self) {
        if !self.enabled {
            return;
        }
        let total = self.clean + self.untrusted + self.malformed + self.error;
        let mut parts = Vec::new();
        if self.clean > 0 {
            parts.push(self.styler.green(&format!("{} clean", self.clean)));
        }
        if self.untrusted > 0 {
            parts.push(self.styler.red(&format!("{} untrusted", self.untrusted)));
        }
        if self.malformed > 0 {
            parts.push(self.styler.yellow(&format!("{} malformed", self.malformed)));
        }
        if self.error > 0 {
            parts.push(self.styler.red(&format!("{} error", self.error)));
        }
        let breakdown = if parts.is_empty() {
            String::new()
        } else {
            format!(" — {}", parts.join(", "))
        };
        eprintln!(
            "{} {} artifact(s){}, {} in {}",
            self.styler.bold("✓ scanned"),
            total,
            breakdown,
            human_size(self.bytes_total),
            human_duration(self.overall_start.elapsed())
        );
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

fn human_duration(d: std::time::Duration) -> String {
    let ms = d.as_millis();
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}
