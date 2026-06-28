//! GGUF metadata sanity + offset validation.
//!
//! GGUF carries no executable code, but its metadata can smuggle a Jinja2 chat
//! template — a code-ish injection surface — so we surface those for human
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
        Ok(()) => {
            if report.verdict != Verdict::Malformed {
                // Clean unless something downgraded it (e.g. chat template -> still
                // not "untrusted", just a low finding for review).
                report.verdict = Verdict::Clean;
            }
        }
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
                    "embedded chat template present — review before trusting (Jinja2 \
                     templates are an injection surface)",
                )
                .with_evidence(vec![format!("{key}: {snippet}")]),
            );
        }
    }

    // --- tensor info block ---
    struct TInfo {
        name: String,
        offset: u64,
    }
    let mut tensors: Vec<TInfo> = Vec::new();
    for _ in 0..tensor_count {
        let name = c.gstring()?;
        let n_dims = c.u32()? as usize;
        if n_dims > 8 {
            return Err(format!("tensor '{name}' declares {n_dims} dims (>8)"));
        }
        for _ in 0..n_dims {
            let _dim = c.u64()?;
        }
        let _ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TInfo { name, offset });
    }

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
                .with_evidence(vec![format!("data_start={data_start}, alignment={alignment}")]),
            );
            report.verdict = Verdict::Untrusted;
        }
    }

    Ok(())
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
        let b = Builder::new(3, 0, 1).kv_string("general.name", "test").buf.clone();
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
}
