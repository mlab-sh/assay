//! safetensors structural validation.
//!
//! The format is safe *by design* (no executable code) but still has
//! format-level attack surface: a malformed header or overlapping/out-of-bounds
//! tensor offsets can cause out-of-bounds reads or DoS at load time. We parse
//! the `u64` length prefix + JSON header and validate every tensor's byte range.

use serde_json::Value;

use crate::hash::{self, TensorEntry};
use crate::report::{ArtifactReport, Finding, Severity, Verdict};

/// Byte size of a safetensors dtype, if known.
fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}

pub fn analyze(artifact_name: &str, data: &[u8]) -> ArtifactReport {
    let mut report = ArtifactReport::new(artifact_name, "safetensors");

    if data.len() < 8 {
        return malformed(report, "ST_HEADER_MALFORMED", "file shorter than 8-byte header length");
    }
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let header_end = match 8u64.checked_add(header_len) {
        Some(v) => v,
        None => {
            return malformed(report, "ST_HEADER_MALFORMED", "header length overflows file size")
        }
    };
    if header_end as usize > data.len() {
        return malformed(
            report,
            "ST_HEADER_MALFORMED",
            format!(
                "declared header length {header_len} exceeds file size {}",
                data.len()
            ),
        );
    }

    let header_bytes = &data[8..header_end as usize];
    let header: Value = match serde_json::from_slice(header_bytes) {
        Ok(v) => v,
        Err(e) => {
            return malformed(report, "ST_HEADER_MALFORMED", format!("header is not valid JSON: {e}"))
        }
    };
    let obj = match header.as_object() {
        Some(o) => o,
        None => {
            return malformed(report, "ST_HEADER_MALFORMED", "header JSON is not an object")
        }
    };

    let data_start = header_end as usize;
    let data_seg_len = (data.len() - data_start) as u64;

    // (begin, end, name) for overlap analysis.
    let mut intervals: Vec<(u64, u64, String)> = Vec::new();
    let mut tensor_entries: Vec<TensorEntry> = Vec::new();
    let mut had_structural_finding = false;

    for (name, spec) in obj {
        if name == "__metadata__" {
            continue;
        }
        let spec = match spec.as_object() {
            Some(s) => s,
            None => {
                report.push(Finding::new(
                    "ST_HEADER_MALFORMED",
                    Severity::Medium,
                    format!("tensor '{name}' spec is not an object"),
                ));
                had_structural_finding = true;
                continue;
            }
        };

        let dtype = spec.get("dtype").and_then(|v| v.as_str()).unwrap_or("");
        let shape: Vec<u64> = spec
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        let offsets = spec
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
            .unwrap_or_default();

        if offsets.len() != 2 {
            report.push(Finding::new(
                "ST_HEADER_MALFORMED",
                Severity::Medium,
                format!("tensor '{name}' has malformed data_offsets"),
            ));
            had_structural_finding = true;
            continue;
        }
        let (begin, end) = (offsets[0], offsets[1]);

        // Offset sanity.
        if begin > end {
            report.push(Finding::new(
                "ST_OFFSET_OOB",
                Severity::High,
                format!("tensor '{name}' has begin ({begin}) > end ({end})"),
            ));
            had_structural_finding = true;
            continue;
        }
        if end > data_seg_len {
            report.push(
                Finding::new(
                    "ST_OFFSET_OOB",
                    Severity::High,
                    format!("tensor '{name}' end offset {end} exceeds data segment ({data_seg_len} bytes)"),
                )
                .with_evidence(vec![format!("data_offsets [{begin}, {end}]")]),
            );
            had_structural_finding = true;
            continue;
        }

        // dtype/shape vs declared byte range.
        let span = end - begin;
        match dtype_size(dtype) {
            Some(sz) => {
                let elems: u128 = shape.iter().map(|&d| d as u128).product::<u128>().max(1);
                let expected = elems * sz as u128;
                if expected != span as u128 {
                    report.push(Finding::new(
                        "ST_DTYPE_SHAPE_MISMATCH",
                        Severity::Medium,
                        format!(
                            "tensor '{name}': dtype {dtype} shape {shape:?} implies {expected} bytes but range spans {span}"
                        ),
                    ));
                    had_structural_finding = true;
                }
            }
            None => {
                report.push(Finding::new(
                    "ST_DTYPE_UNKNOWN",
                    Severity::Low,
                    format!("tensor '{name}' has unknown dtype '{dtype}'"),
                ));
            }
        }

        // Per-tensor digest over the actual bytes.
        let digest = hash::blake3_hex(&data[data_start + begin as usize..data_start + end as usize]);
        report
            .hashes
            .per_tensor
            .insert(name.clone(), hash::tagged(&digest));
        tensor_entries.push(TensorEntry {
            name: name.clone(),
            dtype: dtype.to_string(),
            shape,
            digest,
        });

        intervals.push((begin, end, name.clone()));
    }

    // Overlap detection.
    intervals.sort_by_key(|t| t.0);
    for w in intervals.windows(2) {
        let (_, end_a, name_a) = &w[0];
        let (begin_b, _, name_b) = &w[1];
        if begin_b < end_a {
            report.push(
                Finding::new(
                    "ST_OFFSET_OVERLAP",
                    Severity::High,
                    format!("tensors '{name_a}' and '{name_b}' have overlapping byte ranges"),
                )
                .with_evidence(vec![format!(
                    "'{name_a}' ends at {end_a}, '{name_b}' begins at {begin_b}"
                )]),
            );
            had_structural_finding = true;
        }
    }

    // Manifest hash (stable across rename/repack).
    if !tensor_entries.is_empty() {
        report.hashes.manifest = Some(hash::manifest_hash(&mut tensor_entries));
    }

    report.verdict = if had_structural_finding {
        Verdict::Untrusted
    } else {
        Verdict::Clean
    };
    report
}

fn malformed(
    mut report: ArtifactReport,
    id: &str,
    detail: impl Into<String>,
) -> ArtifactReport {
    report.verdict = Verdict::Malformed;
    report.push(Finding::new(id, Severity::High, detail));
    report
}

// ---------------------------------------------------------------------------
// Phase 2 support: tensor extraction (read-only, does not affect verdicts).
// ---------------------------------------------------------------------------

/// A tensor located within the file's data segment.
pub struct StTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute byte offset/length within the file.
    pub offset: usize,
    pub len: usize,
}

pub struct StExtract {
    pub tensors: Vec<StTensor>,
    pub metadata: std::collections::BTreeMap<String, String>,
}

/// Parse the header and locate every tensor's byte range. Returns `Err` on a
/// structurally unreadable header (the Phase 1 `analyze` already reports why).
pub fn extract(data: &[u8]) -> Result<StExtract, String> {
    if data.len() < 8 {
        return Err("file shorter than header length".into());
    }
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or("header length overflow")? as usize;
    if header_end > data.len() {
        return Err("header length exceeds file".into());
    }
    let header: Value =
        serde_json::from_slice(&data[8..header_end]).map_err(|e| format!("bad header json: {e}"))?;
    let obj = header.as_object().ok_or("header is not an object")?;

    let data_start = header_end;
    let data_seg_len = data.len() - data_start;
    let mut tensors = Vec::new();
    let mut metadata = std::collections::BTreeMap::new();

    for (name, spec) in obj {
        if name == "__metadata__" {
            if let Some(m) = spec.as_object() {
                for (k, v) in m {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }
        let spec = match spec.as_object() {
            Some(s) => s,
            None => continue,
        };
        let dtype = spec
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let shape: Vec<u64> = spec
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        let offsets = spec
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
            .unwrap_or_default();
        if offsets.len() != 2 {
            continue;
        }
        let (begin, end) = (offsets[0], offsets[1]);
        if begin > end || end > data_seg_len as u64 {
            continue; // out-of-bounds tensors are flagged by Phase 1; skip here
        }
        tensors.push(StTensor {
            name: name.clone(),
            dtype,
            shape,
            offset: data_start + begin as usize,
            len: (end - begin) as usize,
        });
    }

    Ok(StExtract { tensors, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal safetensors buffer from a header object + data bytes.
    fn build(header: &Value, data: &[u8]) -> Vec<u8> {
        let hdr = serde_json::to_vec(header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn clean_file_is_clean_and_hashed() {
        let header = serde_json::json!({
            "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
        });
        let buf = build(&header, &[1, 2, 3, 4]);
        let r = analyze("model.safetensors", &buf);
        assert_eq!(r.verdict, Verdict::Clean);
        assert!(r.hashes.manifest.is_some());
    }

    #[test]
    fn rename_does_not_change_manifest() {
        let header = serde_json::json!({
            "weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
        });
        let buf = build(&header, &[1, 2, 3, 4]);
        let a = analyze("a.safetensors", &buf);
        let b = analyze("renamed.safetensors", &buf);
        assert_eq!(a.hashes.manifest, b.hashes.manifest);
    }

    #[test]
    fn overlapping_offsets_flagged() {
        let header = serde_json::json!({
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]}
        });
        let buf = build(&header, &[0u8; 6]);
        let r = analyze("bad.safetensors", &buf);
        assert!(r.findings.iter().any(|f| f.id == "ST_OFFSET_OVERLAP"));
        assert_eq!(r.verdict, Verdict::Untrusted);
    }

    #[test]
    fn oversized_header_is_malformed() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(9999u64).to_le_bytes());
        buf.extend_from_slice(b"{}");
        let r = analyze("bad.safetensors", &buf);
        assert_eq!(r.verdict, Verdict::Malformed);
    }
}
