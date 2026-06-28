//! 2c — secret & string scanning over metadata, GGUF KV blocks, and sibling
//! config/tokenizer files. Extends the Phase 1 chat-template flag into a
//! general scan. Plus an **experimental, opt-in** high-entropy tensor-region
//! check (noisy by nature — gated behind a flag and clearly labeled).

use std::path::Path;

use crate::report::{Finding, Severity};

/// Sibling text files worth scanning when a directory is given.
const SIBLING_FILES: [&str; 6] = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "model_index.json",
];

const MAX_SIBLING_BYTES: u64 = 4 * 1024 * 1024;

/// Scan a set of `(source, text)` pairs plus sibling config files for secrets
/// and suspicious URLs. Findings are returned sorted for determinism.
pub fn scan(strings: &[(String, String)], artifact_dir: Option<&Path>) -> Vec<Finding> {
    let mut findings = Vec::new();

    for (src, text) in strings {
        scan_text(src, text, &mut findings);
    }

    if let Some(dir) = artifact_dir {
        for name in SIBLING_FILES {
            let p = dir.join(name);
            if let Ok(meta) = std::fs::metadata(&p) {
                if meta.is_file() && meta.len() <= MAX_SIBLING_BYTES {
                    if let Ok(content) = std::fs::read_to_string(&p) {
                        scan_text(name, &content, &mut findings);
                    }
                }
            }
        }
    }

    findings.sort_by(|a, b| (a.id.clone(), a.detail.clone()).cmp(&(b.id.clone(), b.detail.clone())));
    findings.dedup_by(|a, b| a.id == b.id && a.detail == b.detail);
    findings
}

fn scan_text(source: &str, text: &str, findings: &mut Vec<Finding>) {
    // URLs.
    for url in extract_urls(text) {
        findings.push(
            Finding::new(
                "SUSPICIOUS_URL",
                Severity::Info,
                format!("external URL referenced in {source}"),
            )
            .with_evidence(vec![truncate(&url, 120)]),
        );
    }

    // Known secret patterns.
    for tok in tokenize(text) {
        if let Some((label, conf)) = classify_secret(tok) {
            let sev = match conf {
                Confidence::High => Severity::High,
                Confidence::Medium => Severity::Medium,
                Confidence::Low => Severity::Low,
            };
            findings.push(
                Finding::new(
                    "EMBEDDED_SECRET",
                    sev,
                    format!("possible {label} in {source}"),
                )
                .with_evidence(vec![redact(tok)]),
            );
        }
    }

    if text.contains("-----BEGIN") && text.contains("PRIVATE KEY-----") {
        findings.push(Finding::new(
            "EMBEDDED_SECRET",
            Severity::High,
            format!("PEM private key block in {source}"),
        ));
    }
}

enum Confidence {
    High,
    Medium,
    Low,
}

fn classify_secret(tok: &str) -> Option<(&'static str, Confidence)> {
    let len = tok.len();
    if tok.starts_with("AKIA") && len == 20 && tok[4..].bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()) {
        return Some(("AWS access key id", Confidence::High));
    }
    if (tok.starts_with("ghp_") || tok.starts_with("github_pat_")) && len >= 36 {
        return Some(("GitHub token", Confidence::High));
    }
    if tok.starts_with("xoxb-") || tok.starts_with("xoxp-") {
        return Some(("Slack token", Confidence::High));
    }
    if tok.starts_with("sk-") && len >= 20 {
        return Some(("API secret key", Confidence::Medium));
    }
    if tok.starts_with("AIza") && len >= 35 {
        return Some(("Google API key", Confidence::Medium));
    }
    // Generic high-entropy blob (low confidence; easy false positives).
    if len >= 32 && looks_tokenish(tok) && shannon_entropy_str(tok) > 4.0 {
        return Some(("high-entropy token", Confidence::Low));
    }
    None
}

fn looks_tokenish(s: &str) -> bool {
    s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'+' || b == b'/' || b == b'=')
}

fn tokenize(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+' | '/' | '=')))
        .filter(|t| t.len() >= 8)
}

fn extract_urls(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for scheme in ["https://", "http://"] {
        let mut rest = text;
        while let Some(pos) = rest.find(scheme) {
            let tail = &rest[pos..];
            let end = tail
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '<' || c == ')')
                .unwrap_or(tail.len());
            out.push(tail[..end].to_string());
            rest = &tail[end.max(1)..];
        }
    }
    out.sort();
    out.dedup();
    out
}

fn redact(tok: &str) -> String {
    let n = tok.len();
    if n <= 8 {
        "*".repeat(n)
    } else {
        format!("{}…{} ({} chars)", &tok[..4], &tok[n - 2..], n)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

fn shannon_entropy_str(s: &str) -> f64 {
    let bytes = s.as_bytes();
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    let mut h = 0.0;
    for &c in counts.iter() {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

/// Shannon entropy of a byte buffer (bits/byte). Shared with the experimental
/// tensor-entropy check.
pub fn shannon_entropy_bytes(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    let mut h = 0.0;
    for &c in counts.iter() {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_url_and_token() {
        let strings = vec![(
            "config.json".to_string(),
            "see https://evil.example.com/x and key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".to_string(),
        )];
        let f = scan(&strings, None);
        assert!(f.iter().any(|x| x.id == "SUSPICIOUS_URL"));
        assert!(f.iter().any(|x| x.id == "EMBEDDED_SECRET"));
    }

    #[test]
    fn clean_text_is_quiet() {
        let strings = vec![("config.json".to_string(), "hidden_size: 768, num_heads: 12".to_string())];
        assert!(scan(&strings, None).is_empty());
    }
}
