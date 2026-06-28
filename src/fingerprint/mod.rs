//! 2d — architectural fingerprint.
//!
//! Derive a cheap structural signature (naming scheme, layer count, hidden dim,
//! heads, vocab) and compare it against the declared identity. Catches
//! "claims to be X but is structurally Y" masked-repackaging. Heuristic by
//! nature — emits the detected family as info, and `ARCH_MISMATCH` only on a
//! clear naming-scheme/declared-architecture contradiction.

use serde::Serialize;

use crate::report::{Finding, Severity};

#[derive(Debug, Clone, Serialize)]
pub struct Fingerprint {
    pub scheme: String,
    pub detected_family: Option<String>,
    pub declared: Option<String>,
    pub layer_count: Option<u64>,
    pub hidden: Option<u64>,
    pub n_heads: Option<u64>,
    pub vocab: Option<u64>,
}

pub struct FpInput {
    pub tensor_names: Vec<String>,
    pub declared_arch: Option<String>,
    pub layer_count: Option<u64>,
    pub hidden: Option<u64>,
    pub n_heads: Option<u64>,
    pub vocab: Option<u64>,
}

/// Detect the tensor naming scheme.
fn naming_scheme(names: &[String]) -> &'static str {
    let has = |needle: &str| names.iter().any(|n| n.contains(needle));
    if has("blk.") {
        "gguf-blk"
    } else if has("model.layers.") {
        "hf-llama"
    } else if has("transformer.h.") || has("h.") {
        "gpt2"
    } else {
        "unknown"
    }
}

/// Map (scheme, declared arch) to a family guess.
fn family_from_scheme(scheme: &str, declared: Option<&str>) -> Option<String> {
    let d = declared.unwrap_or("").to_ascii_lowercase();
    let fam = match scheme {
        "hf-llama" | "gguf-blk" => {
            if d.contains("qwen") {
                "qwen"
            } else if d.contains("mistral") {
                "mistral"
            } else if d.contains("gemma") {
                "gemma"
            } else if d.contains("phi") {
                "phi"
            } else if d.contains("llama") {
                "llama"
            } else {
                "llama-family"
            }
        }
        "gpt2" => "gpt2",
        _ => return None,
    };
    Some(fam.to_string())
}

/// Does the declared architecture clearly contradict the naming scheme?
fn is_mismatch(scheme: &str, declared: &str) -> bool {
    let d = declared.to_ascii_lowercase();
    match scheme {
        "gpt2" => !(d.contains("gpt2") || d.contains("gpt-2") || d.is_empty()),
        "hf-llama" | "gguf-blk" => {
            // gpt2 declared but llama-style names -> contradiction.
            d.contains("gpt2") || d.contains("gpt-2") || d.contains("bert")
        }
        _ => false,
    }
}

pub fn analyze(input: &FpInput) -> (Fingerprint, Vec<Finding>) {
    let scheme = naming_scheme(&input.tensor_names);
    let detected_family = family_from_scheme(scheme, input.declared_arch.as_deref());

    let fp = Fingerprint {
        scheme: scheme.to_string(),
        detected_family: detected_family.clone(),
        declared: input.declared_arch.clone(),
        layer_count: input.layer_count,
        hidden: input.hidden,
        n_heads: input.n_heads,
        vocab: input.vocab,
    };

    let mut findings = Vec::new();

    if let Some(declared) = &input.declared_arch {
        if scheme != "unknown" && is_mismatch(scheme, declared) {
            findings.push(
                Finding::new(
                    "ARCH_MISMATCH",
                    Severity::Medium,
                    format!(
                        "declared architecture '{declared}' disagrees with the structural \
                         signature ({scheme}) — possible masked repackaging"
                    ),
                )
                .with_evidence(vec![format!(
                    "scheme={scheme}, declared={declared}, detected_family={}",
                    detected_family.clone().unwrap_or_else(|| "?".into())
                )]),
            );
        }
    }

    if findings.is_empty() {
        let fam = detected_family.clone().unwrap_or_else(|| "unknown".into());
        findings.push(
            Finding::new(
                "ARCH_DETECTED",
                Severity::Info,
                format!("structural fingerprint: {fam} ({scheme})"),
            )
            .with_evidence(vec![format!(
                "layers={:?}, hidden={:?}, heads={:?}, vocab={:?}",
                input.layer_count, input.hidden, input.n_heads, input.vocab
            )]),
        );
    }

    (fp, findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mismatch() {
        let input = FpInput {
            tensor_names: vec!["model.layers.0.self_attn.q_proj.weight".into()],
            declared_arch: Some("gpt2".into()),
            layer_count: Some(32),
            hidden: Some(4096),
            n_heads: Some(32),
            vocab: Some(32000),
        };
        let (_fp, f) = analyze(&input);
        assert!(f.iter().any(|x| x.id == "ARCH_MISMATCH"));
    }

    #[test]
    fn consistent_emits_info() {
        let input = FpInput {
            tensor_names: vec!["model.layers.0.self_attn.q_proj.weight".into()],
            declared_arch: Some("llama".into()),
            layer_count: Some(32),
            hidden: Some(4096),
            n_heads: Some(32),
            vocab: Some(32000),
        };
        let (_fp, f) = analyze(&input);
        assert!(f.iter().any(|x| x.id == "ARCH_DETECTED"));
        assert!(!f.iter().any(|x| x.id == "ARCH_MISMATCH"));
    }
}
