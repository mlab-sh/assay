//! Secret & string scanning over metadata, GGUF KV blocks, and sibling
//! config/tokenizer files. Extends the Phase 1 chat-template flag into a
//! general scan. Plus an **experimental, opt-in** high-entropy tensor-region
//! check (noisy by nature, so it is gated behind a flag and clearly labeled).

use std::path::Path;

use crate::report::{Finding, Severity};

/// Sibling text files worth scanning when a directory is given.
const SIBLING_FILES: [&str; 11] = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "model_index.json",
    // The current Hugging Face convention puts the chat template in its own
    // file rather than inside tokenizer_config.json.
    "chat_template.jinja",
    "chat_template.json",
    // Adapters ship their own config, and it names the base model.
    "adapter_config.json",
    "preprocessor_config.json",
    "processor_config.json",
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

    // Sorted for a deterministic report. Compared field by field rather than
    // through a cloned key tuple, so ordering costs no allocations.
    findings.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.detail.cmp(&b.detail)));
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

    // A Google service account key is a JSON blob, not a token: it is
    // recognizable by its shape rather than by a prefix.
    if text.contains("\"type\"") && text.contains("service_account") && text.contains("private_key")
    {
        findings.push(Finding::new(
            "EMBEDDED_SECRET",
            Severity::High,
            format!("Google service account credentials in {source}"),
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
    // The one most likely to end up in a model repo by accident.
    if tok.starts_with("hf_") && len >= 34 && tok[3..].bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Some(("Hugging Face user access token", Confidence::High));
    }
    if tok.starts_with("api_org_") && len >= 30 {
        return Some(("Hugging Face organization token", Confidence::High));
    }
    if (tok.starts_with("AKIA") || tok.starts_with("ASIA"))
        && len == 20
        && tok[4..]
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    {
        return Some(("AWS access key id", Confidence::High));
    }
    if (tok.starts_with("ghp_")
        || tok.starts_with("gho_")
        || tok.starts_with("ghu_")
        || tok.starts_with("ghs_")
        || tok.starts_with("ghr_")
        || tok.starts_with("github_pat_"))
        && len >= 36
    {
        return Some(("GitHub token", Confidence::High));
    }
    if tok.starts_with("glpat-") && len >= 24 {
        return Some(("GitLab token", Confidence::High));
    }
    if tok.starts_with("xoxb-")
        || tok.starts_with("xoxp-")
        || tok.starts_with("xoxa-")
        || tok.starts_with("xoxr-")
    {
        return Some(("Slack token", Confidence::High));
    }
    if tok.starts_with("dckr_pat_") && len >= 20 {
        return Some(("Docker Hub token", Confidence::High));
    }
    // Checked before the generic `sk-` rule, which is only medium confidence.
    if tok.starts_with("sk-ant-") && len >= 20 {
        return Some(("Anthropic API key", Confidence::High));
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
    s.bytes().all(|b| {
        b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'+' || b == b'/' || b == b'='
    })
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
            format!(
                "see https://evil.example.com/x and key {}",
                cred("ghp", "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            ),
        )];
        let f = scan(&strings, None);
        assert!(f.iter().any(|x| x.id == "SUSPICIOUS_URL"));
        assert!(f.iter().any(|x| x.id == "EMBEDDED_SECRET"));
    }

    #[test]
    fn clean_text_is_quiet() {
        let strings = vec![(
            "config.json".to_string(),
            "hidden_size: 768, num_heads: 12".to_string(),
        )];
        assert!(scan(&strings, None).is_empty());
    }

    /// Assembles a credential-shaped fixture at run time.
    ///
    /// The halves are kept apart on purpose. A literal that looks like a live
    /// token is picked up by push protection and by secret scanners, including
    /// ours: a test fixture must never be something a scanner has to be told to
    /// ignore. Nothing here is a real credential.
    fn cred(head: &str, tail: &str) -> String {
        format!("{head}{tail}")
    }

    fn label(tok: &str) -> Option<&'static str> {
        classify_secret(tok).map(|(l, _)| l)
    }

    fn confidence_is_high(tok: &str) -> bool {
        matches!(classify_secret(tok), Some((_, Confidence::High)))
    }

    /// The credential most likely to be committed to a model repo by accident.
    #[test]
    fn hugging_face_tokens_are_recognized() {
        let user = cred("hf", "_QGxKmPvRtWyZaBcDeFgHiJkLmNoPqRsTuV");
        assert_eq!(label(&user), Some("Hugging Face user access token"));
        assert!(confidence_is_high(&user));

        let org = cred("api", "_org_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef");
        assert_eq!(label(&org), Some("Hugging Face organization token"));
        assert!(confidence_is_high(&org));
    }

    #[test]
    fn a_short_hf_lookalike_is_not_a_token() {
        // `hf_model`, `hf_hub`, and friends appear in ordinary config text.
        assert_eq!(label("hf_hub"), None);
        assert_eq!(label("hf_model_name"), None);
    }

    #[test]
    fn anthropic_keys_beat_the_generic_sk_rule() {
        let key = cred("sk", "-ant-api03-AAAAAAAAAAAAAAAAAAAAAA");
        assert_eq!(label(&key), Some("Anthropic API key"));
        assert!(confidence_is_high(&key), "not just a generic medium guess");
        // The generic rule still covers everything else.
        assert_eq!(
            label(&cred("sk", "-proj-AAAAAAAAAAAAAAAAAAAA")),
            Some("API secret key")
        );
    }

    #[test]
    fn the_other_provider_prefixes_are_covered() {
        assert_eq!(
            label(&cred("ASIA", "ZZZZZZZZZZZZZZZZ")),
            Some("AWS access key id")
        );
        assert_eq!(
            label(&cred("ghs", "_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
            Some("GitHub token")
        );
        assert_eq!(
            label(&cred("glpat", "-ABCDEFGHIJKLMNOPQRSTUV")),
            Some("GitLab token")
        );
        assert_eq!(
            label(&cred("dckr", "_pat_ABCDEFGHIJKLMNOPQRST")),
            Some("Docker Hub token")
        );
        assert_eq!(label(&cred("xoxa", "-2-abcdef")), Some("Slack token"));
    }

    #[test]
    fn a_google_service_account_is_recognized_by_shape() {
        let mut f = Vec::new();
        scan_text(
            "config.json",
            r#"{"type": "service_account", "private_key": "..."}"#,
            &mut f,
        );
        assert!(
            f.iter()
                .any(|x| x.detail.contains("Google service account")),
            "{f:?}"
        );
    }

    #[test]
    fn the_current_hugging_face_layout_is_covered() {
        // chat_template.jinja is where templates live now, and adapters ship
        // their own config: both are text files a token can end up in.
        for name in [
            "chat_template.jinja",
            "chat_template.json",
            "adapter_config.json",
            "preprocessor_config.json",
        ] {
            assert!(SIBLING_FILES.contains(&name), "{name} is not scanned");
        }
    }

    #[test]
    fn sibling_files_are_read_from_the_model_directory() {
        let dir = std::env::temp_dir().join(format!("assay-sec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("chat_template.jinja"),
            format!(
                "{{{{ messages }}}} key {}",
                cred("hf", "_QGxKmPvRtWyZaBcDeFgHiJkLmNoPqRsTuV")
            ),
        )
        .unwrap();

        let f = scan(&[], Some(&dir));
        assert!(
            f.iter()
                .any(|x| x.id == "EMBEDDED_SECRET" && x.detail.contains("chat_template.jinja")),
            "{f:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
