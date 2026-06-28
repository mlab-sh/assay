//! Signature / provenance verification — the honest Phase-1 subset.
//!
//! We verify only what can be verified offline and with high confidence:
//!   * an OpenSSF model-transparency style manifest that records the expected
//!     manifest hash (compared against what we computed), and
//!   * a detached ed25519 signature over the manifest hash, against a public
//!     key the caller supplies.
//!
//! Full Sigstore / cosign verification (Fulcio certificate chain + Rekor
//! transparency log) is **not** implemented yet. When we see such a bundle we
//! say so plainly rather than implying trust — per the README's "honest
//! confidence" principle, `signed` is only ever reported on a real pass.

use std::path::{Path, PathBuf};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::report::{Finding, Severity};

pub struct SignatureOutcome {
    /// Goes into the report's `signature` field: e.g. "unsigned", "signed",
    /// "signature-mismatch", "unverified (sigstore)".
    pub status: String,
    pub findings: Vec<Finding>,
}

impl SignatureOutcome {
    fn unsigned() -> Self {
        SignatureOutcome {
            status: "unsigned".into(),
            findings: Vec::new(),
        }
    }
}

/// Evaluate signature material for an artifact.
///
/// * `computed_manifest` — the manifest hash `assay` computed (`blake3:…`).
/// * `bundle` — explicit bundle/signature path (from `verify --bundle`), else
///   we auto-detect common sidecars next to the artifact.
/// * `key` — explicit ed25519 public-key path (from `verify --key`).
pub fn evaluate(
    artifact_path: &Path,
    computed_manifest: Option<&str>,
    bundle: Option<&Path>,
    key: Option<&Path>,
) -> SignatureOutcome {
    let bundle_path = match bundle.map(PathBuf::from).or_else(|| autodetect(artifact_path)) {
        Some(p) => p,
        None => return SignatureOutcome::unsigned(),
    };

    let raw = match std::fs::read(&bundle_path) {
        Ok(b) => b,
        Err(e) => {
            return SignatureOutcome {
                status: "unsigned".into(),
                findings: vec![Finding::new(
                    "SIG_BUNDLE_UNREADABLE",
                    Severity::Low,
                    format!("found {} but could not read it: {e}", bundle_path.display()),
                )],
            }
        }
    };
    let display = bundle_path.display().to_string();

    // Try to interpret the bundle as JSON first.
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&raw) {
        if is_sigstore_bundle(&json) {
            return sigstore_unverified(&display);
        }
        if let Some(expected) = json.get("manifest").and_then(|v| v.as_str()) {
            return verify_transparency(expected, computed_manifest, &display);
        }
        // JSON we don't recognize — be honest.
        return SignatureOutcome {
            status: "unverified".into(),
            findings: vec![Finding::new(
                "SIG_UNRECOGNIZED",
                Severity::Low,
                format!("{display} is a JSON bundle in an unrecognized format; not verified"),
            )],
        };
    }

    // Filename hints for a Sigstore bundle even if not JSON-parseable here.
    let lower = display.to_ascii_lowercase();
    if lower.ends_with(".sigstore") || lower.ends_with(".bundle") {
        return sigstore_unverified(&display);
    }

    // Otherwise treat it as a detached ed25519 signature over the manifest hash.
    verify_detached(&raw, computed_manifest, key, &display)
}

fn autodetect(artifact_path: &Path) -> Option<PathBuf> {
    let candidates = [
        format!("{}.sig", artifact_path.display()),
        format!("{}.sigstore", artifact_path.display()),
        format!("{}.bundle", artifact_path.display()),
        format!("{}.manifest.json", artifact_path.display()),
    ];
    for c in candidates {
        let p = PathBuf::from(&c);
        if p.exists() {
            return Some(p);
        }
    }
    // model-transparency convention: a `model.sig` in the same directory.
    if let Some(dir) = artifact_path.parent() {
        let p = dir.join("model.sig");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn is_sigstore_bundle(json: &serde_json::Value) -> bool {
    json.get("verificationMaterial").is_some()
        || json.get("messageSignature").is_some()
        || json.get("tlogEntries").is_some()
        || json
            .get("mediaType")
            .and_then(|v| v.as_str())
            .map(|s| s.contains("sigstore"))
            .unwrap_or(false)
}

fn sigstore_unverified(display: &str) -> SignatureOutcome {
    SignatureOutcome {
        status: "unverified (sigstore)".into(),
        findings: vec![Finding::new(
            "SIG_SIGSTORE_UNVERIFIED",
            Severity::Info,
            "a Sigstore/cosign bundle is present, but full chain verification \
             (Fulcio cert + Rekor log) is not implemented in Phase 1; treat as unverified",
        )
        .with_evidence(vec![display.to_string()])],
    }
}

fn verify_transparency(
    expected: &str,
    computed: Option<&str>,
    display: &str,
) -> SignatureOutcome {
    match computed {
        Some(c) if c == expected => SignatureOutcome {
            status: "signed".into(),
            findings: vec![Finding::new(
                "SIG_MANIFEST_MATCH",
                Severity::Info,
                "manifest hash matches the recorded transparency manifest",
            )
            .with_evidence(vec![display.to_string()])],
        },
        Some(c) => SignatureOutcome {
            status: "signature-mismatch".into(),
            findings: vec![Finding::new(
                "SIG_MISMATCH",
                Severity::High,
                "computed manifest hash does NOT match the recorded transparency manifest",
            )
            .with_evidence(vec![format!("expected {expected}, computed {c}")])],
        },
        None => SignatureOutcome {
            status: "unverified".into(),
            findings: vec![Finding::new(
                "SIG_NO_MANIFEST",
                Severity::Low,
                "transparency manifest present but no manifest hash was computed to compare",
            )],
        },
    }
}

fn verify_detached(
    sig_bytes: &[u8],
    computed: Option<&str>,
    key: Option<&Path>,
    display: &str,
) -> SignatureOutcome {
    let key_path = match key {
        Some(k) => k,
        None => {
            return SignatureOutcome {
                status: "unverified".into(),
                findings: vec![Finding::new(
                    "SIG_NO_KEY",
                    Severity::Info,
                    "detached signature present but no public key supplied (--key); not verified",
                )
                .with_evidence(vec![display.to_string()])],
            }
        }
    };

    let msg = match computed {
        Some(m) => m.as_bytes().to_vec(),
        None => {
            return SignatureOutcome {
                status: "unverified".into(),
                findings: vec![Finding::new(
                    "SIG_NO_MANIFEST",
                    Severity::Low,
                    "signature present but no manifest hash was computed to verify against",
                )],
            }
        }
    };

    let key_raw = match std::fs::read(key_path) {
        Ok(b) => b,
        Err(e) => {
            return SignatureOutcome {
                status: "unverified".into(),
                findings: vec![Finding::new(
                    "SIG_KEY_UNREADABLE",
                    Severity::Low,
                    format!("could not read key {}: {e}", key_path.display()),
                )],
            }
        }
    };

    let vk_bytes = match parse_fixed::<32>(&key_raw) {
        Some(b) => b,
        None => return sig_error("public key is not 32 raw or hex-encoded bytes"),
    };
    let sig_fixed = match parse_fixed::<64>(sig_bytes) {
        Some(b) => b,
        None => return sig_error("signature is not 64 raw or hex-encoded bytes"),
    };

    let vk = match VerifyingKey::from_bytes(&vk_bytes) {
        Ok(v) => v,
        Err(e) => return sig_error(&format!("invalid ed25519 public key: {e}")),
    };
    let sig = Signature::from_bytes(&sig_fixed);

    match vk.verify(&msg, &sig) {
        Ok(()) => SignatureOutcome {
            status: "signed".into(),
            findings: vec![Finding::new(
                "SIG_VERIFIED",
                Severity::Info,
                "detached ed25519 signature over the manifest hash verified",
            )
            .with_evidence(vec![display.to_string()])],
        },
        Err(_) => SignatureOutcome {
            status: "signature-mismatch".into(),
            findings: vec![Finding::new(
                "SIG_MISMATCH",
                Severity::High,
                "detached ed25519 signature failed verification against the supplied key",
            )],
        },
    }
}

fn sig_error(detail: &str) -> SignatureOutcome {
    SignatureOutcome {
        status: "unverified".into(),
        findings: vec![Finding::new("SIG_ERROR", Severity::Low, detail.to_string())],
    }
}

/// Accept either exactly N raw bytes, or a hex string decoding to N bytes.
fn parse_fixed<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
    if bytes.len() == N {
        return bytes.try_into().ok();
    }
    let s = std::str::from_utf8(bytes).ok()?.trim();
    let decoded = hex::decode(s).ok()?;
    if decoded.len() == N {
        decoded.try_into().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn detached_signature_roundtrip() {
        let dir = std::env::temp_dir().join(format!("assay-sig-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = "blake3:deadbeef";

        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let sig = sk.sign(manifest.as_bytes());
        let key_path = dir.join("key.pub");
        let sig_path = dir.join("artifact.safetensors.sig");
        std::fs::write(&key_path, sk.verifying_key().to_bytes()).unwrap();
        std::fs::write(&sig_path, sig.to_bytes()).unwrap();

        let outcome = verify_detached(
            &sig.to_bytes(),
            Some(manifest),
            Some(&key_path),
            "artifact.safetensors.sig",
        );
        assert_eq!(outcome.status, "signed");

        // Wrong manifest -> mismatch.
        let bad = verify_detached(
            &sig.to_bytes(),
            Some("blake3:0000"),
            Some(&key_path),
            "artifact.safetensors.sig",
        );
        assert_eq!(bad.status, "signature-mismatch");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transparency_match_and_mismatch() {
        let ok = verify_transparency("blake3:abc", Some("blake3:abc"), "m.json");
        assert_eq!(ok.status, "signed");
        let no = verify_transparency("blake3:abc", Some("blake3:def"), "m.json");
        assert_eq!(no.status, "signature-mismatch");
    }
}
