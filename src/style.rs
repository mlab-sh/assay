//! Tiny zero-dependency ANSI styling. Color is opt-in per stream: callers
//! decide whether it's enabled (TTY + `NO_COLOR` + `--color` are resolved in
//! `main`), so the same helpers work for the stdout report and stderr progress.

use crate::report::{Severity, Verdict};

#[derive(Clone, Copy)]
pub struct Styler {
    enabled: bool,
}

impl Styler {
    pub fn new(enabled: bool) -> Self {
        Styler { enabled }
    }

    fn wrap(&self, codes: &str, s: &str) -> String {
        if self.enabled {
            format!("\x1b[{codes}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn bold(&self, s: &str) -> String {
        self.wrap("1", s)
    }
    pub fn dim(&self, s: &str) -> String {
        self.wrap("2", s)
    }
    pub fn red(&self, s: &str) -> String {
        self.wrap("31", s)
    }
    pub fn green(&self, s: &str) -> String {
        self.wrap("32", s)
    }
    pub fn yellow(&self, s: &str) -> String {
        self.wrap("33", s)
    }
    pub fn cyan(&self, s: &str) -> String {
        self.wrap("36", s)
    }

    /// Colored severity tag, e.g. `high` in red.
    pub fn severity(&self, sev: Severity) -> String {
        let s = sev.as_str();
        match sev {
            Severity::Critical => self.wrap("1;31", s), // bold red
            Severity::High => self.red(s),
            Severity::Medium => self.yellow(s),
            Severity::Low => self.cyan(s),
            Severity::Info => self.dim(s),
        }
    }

    /// Colored verdict label, e.g. `UNTRUSTED` in red.
    pub fn verdict(&self, v: Verdict) -> String {
        let s = v.as_str().to_uppercase();
        match v {
            Verdict::Clean => self.green(&s),
            Verdict::Untrusted => self.red(&s),
            Verdict::Malformed => self.yellow(&s),
            Verdict::Error => self.wrap("1;31", &s),
        }
    }
}
