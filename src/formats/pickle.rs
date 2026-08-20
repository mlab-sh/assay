//! Pickle / arbitrary-code-execution risk scanner: the highest-priority check.
//!
//! We **never execute** the pickle. We walk the opcode stream statically,
//! resolving global references and noting the opcodes that actually invoke a
//! callable at load time (`REDUCE`, `INST`, `OBJ`, `NEWOBJ`, …). Torch `.bin` /
//! `.pt` files are usually zip containers wrapping one or more pickle streams;
//! we look inside those too.

use std::io::Read;

use crate::report::{ArtifactReport, Finding, Severity, Verdict};

/// Module roots whose mere appearance in a pickle is a strong malicious signal.
const DANGEROUS_MODULES: &[&str] = &[
    "os",
    "posix",
    "nt",
    "subprocess",
    "sys",
    "socket",
    "shutil",
    "pty",
    "ctypes",
    "commands",
    "popen2",
    "webbrowser",
    "importlib",
    "runpy",
    "code",
    "codeop",
    "multiprocessing",
];

/// Specific builtins that enable code execution / attribute pivoting.
const DANGEROUS_BUILTINS: &[&str] = &[
    "eval",
    "exec",
    "execfile",
    "compile",
    "getattr",
    "setattr",
    "__import__",
    "globals",
    "vars",
    "open",
    "input",
];

const EVIDENCE_CAP: usize = 32;

#[derive(Debug)]
struct GlobalRef {
    opcode: &'static str,
    module: String,
    name: String,
}

impl GlobalRef {
    fn qualified(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }

    fn is_dangerous(&self) -> bool {
        let root = self.module.split('.').next().unwrap_or(&self.module);
        if DANGEROUS_MODULES.contains(&root) {
            return true;
        }
        if (root == "builtins" || root == "__builtin__")
            && DANGEROUS_BUILTINS.contains(&self.name.as_str())
        {
            return true;
        }
        false
    }
}

#[derive(Debug, Default)]
struct StreamScan {
    globals: Vec<GlobalRef>,
    /// Opcodes that invoke a callable / mutate an object at load time.
    code_exec_ops: Vec<&'static str>,
    /// The stream stopped before its STOP opcode: analysis is genuinely
    /// incomplete, as opposed to simply having reached the end of a pickle.
    truncated: bool,
    /// Bytes consumed, i.e. one past the STOP opcode for a complete stream.
    end: usize,
    /// First LONG1 value in the stream. The legacy torch container opens with
    /// its magic number encoded this way, which identifies the format exactly.
    first_long1: Option<u128>,
}

/// `torch.save` legacy (non-zip) containers open with this magic, pickled as a
/// LONG1 in a stream of their own.
const TORCH_LEGACY_MAGIC: u128 = 0x1950_a86a_20f9_469c_fc6c;

/// Refuse to walk an unbounded chain of concatenated streams.
const MAX_STREAMS: usize = 64;

/// A raw (non-zip) pickle artifact, split into what it actually contains.
#[derive(Debug, Default)]
struct Container {
    /// One entry per pickle stream, in file order.
    streams: Vec<StreamScan>,
    /// Byte offset where a stream stopped early, if one did.
    truncated_at: Option<usize>,
    /// Bytes after the last complete stream: `(offset, len)`.
    tail: Option<(usize, usize)>,
    /// The first stream carried the legacy `torch.save` magic number.
    torch_legacy: bool,
    /// How many streams were found, and how far opcode analysis reached.
    stream_count: usize,
    analyzed_to: usize,
}

/// Analyze a pickle artifact (raw stream or torch zip container).
pub fn analyze(artifact_name: &str, data: &[u8]) -> ArtifactReport {
    let mut report = ArtifactReport::new(artifact_name, "pickle");
    // A pickle artifact can run code at load time by design, so it is never clean.
    report.verdict = Verdict::Untrusted;

    let mut scans: Vec<(String, StreamScan)> = Vec::new();
    let mut container: Option<Container> = None;

    if data.len() >= 4 && &data[0..4] == b"PK\x03\x04" {
        match collect_zip_pickles(data) {
            Ok(entries) if !entries.is_empty() => {
                for (entry_name, bytes) in entries {
                    scans.push((entry_name, scan_stream(&bytes)));
                }
            }
            Ok(_) => {
                report.push(Finding::new(
                    "PICKLE_CONTAINER_NO_PICKLE",
                    Severity::Medium,
                    "torch zip container had no embedded pickle stream to scan",
                ));
            }
            Err(e) => {
                report.push(Finding::new(
                    "PICKLE_CONTAINER_UNREADABLE",
                    Severity::Medium,
                    format!("could not read torch zip container: {e}"),
                ));
            }
        }
    } else {
        container = Some(split_streams(data));
        if let Some(c) = container.as_mut() {
            let streams = std::mem::take(&mut c.streams);
            let total = streams.len();
            for (i, st) in streams.into_iter().enumerate() {
                let label = if total > 1 {
                    format!("stream {}/{total}", i + 1)
                } else {
                    String::new()
                };
                scans.push((label, st));
            }
        }
    }

    summarize(&mut report, &scans);

    match &container {
        // Raw stream: we know the container layout, so say exactly what was
        // read and what was not.
        Some(c) => describe_container(&mut report, c, data),
        // Zip entries: each is a standalone pickle, so a stream that stopped
        // early really is a truncated pickle.
        None => {
            for (entry, scan) in &scans {
                if scan.truncated {
                    report.push(Finding::new(
                        "PICKLE_TRUNCATED",
                        Severity::Medium,
                        format!(
                            "{}opcode stream stopped at byte {} before its STOP opcode; \
                             analysis of this entry is incomplete",
                            if entry.is_empty() {
                                String::new()
                            } else {
                                format!("{entry}: ")
                            },
                            scan.end
                        ),
                    ));
                }
            }
        }
    }
    report
}

/// Report what the raw container actually held: how far the opcodes went, and
/// what the bytes after them are.
fn describe_container(report: &mut ArtifactReport, c: &Container, data: &[u8]) {
    if let Some(at) = c.truncated_at {
        report.push(Finding::new(
            "PICKLE_TRUNCATED",
            Severity::Medium,
            format!(
                "opcode stream {} stopped at byte {at} of {}, before reaching its STOP \
                 opcode; everything after that offset is unanalyzed",
                c.stream_count,
                data.len()
            ),
        ));
        return;
    }

    let Some((start, len)) = c.tail else {
        return;
    };
    let region = &data[start..];

    if let Some(kind) = crate::magic::identify(region) {
        report.push(
            Finding::new(
                "PICKLE_TRAILING_DATA",
                Severity::High,
                format!(
                    "{len} byte(s) follow the last pickle stream (file offset {start}) and \
                     begin with {kind}; nothing in the container format explains them"
                ),
            )
            .with_evidence(vec![
                format!("file offsets {start}..{}", data.len()),
                format!("first bytes: {}", crate::magic::preview(region)),
            ]),
        );
        return;
    }

    if c.torch_legacy {
        report.push(
            Finding::new(
                "PICKLE_TORCH_LEGACY",
                Severity::Info,
                format!(
                    "legacy torch container: {} pickle stream(s) analyzed in full (bytes \
                     0..{}); the remaining {len} byte(s) are the raw tensor storage payload, \
                     which carries no opcodes",
                    c.stream_count, c.analyzed_to
                ),
            )
            .with_evidence(vec![format!(
                "storage payload at file offsets {start}..{}",
                data.len()
            )]),
        );
        return;
    }

    report.push(
        Finding::new(
            "PICKLE_TRAILING_DATA",
            Severity::Medium,
            format!(
                "{len} byte(s) follow the last pickle stream (file offset {start}) and were \
                 not analyzed; this is not the legacy torch container layout"
            ),
        )
        .with_evidence(vec![format!(
            "first bytes: {}",
            crate::magic::preview(region)
        )]),
    );
}

fn summarize(report: &mut ArtifactReport, scans: &[(String, StreamScan)]) {
    let mut dangerous_evidence: Vec<String> = Vec::new();
    let mut exec_evidence: Vec<String> = Vec::new();
    let mut any_dangerous = false;
    let mut any_exec = false;
    let mut any_global = false;

    for (entry, scan) in scans {
        let prefix = if entry.is_empty() {
            String::new()
        } else {
            format!("{entry}: ")
        };
        if !scan.code_exec_ops.is_empty() {
            any_exec = true;
        }
        for g in &scan.globals {
            any_global = true;
            if g.is_dangerous() {
                any_dangerous = true;
                if dangerous_evidence.len() < EVIDENCE_CAP {
                    dangerous_evidence.push(format!(
                        "{prefix}opcode {} -> {}",
                        g.opcode,
                        g.qualified()
                    ));
                }
            }
        }
        // Summarize the execution triggers seen in this stream.
        let mut seen: Vec<&str> = Vec::new();
        for op in &scan.code_exec_ops {
            if !seen.contains(op) {
                seen.push(op);
            }
        }
        if !seen.is_empty() && exec_evidence.len() < EVIDENCE_CAP {
            exec_evidence.push(format!("{prefix}execution opcodes: {}", seen.join(", ")));
        }
    }

    if any_dangerous || any_exec {
        let mut evidence = dangerous_evidence;
        evidence.extend(exec_evidence);
        report.push(
            Finding::new(
                "PICKLE_RCE_RISK",
                Severity::High,
                "pickle artifact can execute code at load time",
            )
            .with_evidence(evidence),
        );
    } else if any_global {
        report.push(Finding::new(
            "PICKLE_GLOBAL_REF",
            Severity::Medium,
            "pickle references imported globals but no execution opcode was found; \
             loading still deserializes attacker-controlled objects",
        ));
    } else {
        report.push(Finding::new(
            "PICKLE_UNTRUSTED",
            Severity::Medium,
            "pickle format is untrusted by design; no code-execution opcodes detected, \
             but prefer a safetensors equivalent",
        ));
    }
}

fn collect_zip_pickles(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let reader = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".pkl") || lower.ends_with("data.pkl") || lower.ends_with(".pickle") {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            out.push((name, buf));
        }
    }
    Ok(out)
}

/// Walk a single pickle opcode stream without executing it.
fn scan_stream(data: &[u8]) -> StreamScan {
    let mut s = StreamScan::default();
    // Tracks string literals as they are pushed, so STACK_GLOBAL can resolve
    // its (module, name) operands from the two most recent string pushes.
    let mut str_stack: Vec<String> = Vec::new();
    let mut i = 0usize;
    let n = data.len();

    macro_rules! need {
        ($k:expr) => {{
            if i + $k > n {
                s.truncated = true;
                s.end = i;
                return s;
            }
        }};
    }
    // Bounds-check then advance the cursor by $k bytes.
    macro_rules! skip {
        ($k:expr) => {{
            need!($k);
            i += $k;
        }};
    }

    while i < n {
        let op = data[i];
        i += 1;
        match op {
            // STOP: this stream is over. Anything after it is a separate
            // stream or payload, not more opcodes, and reading on is exactly
            // what used to make every legacy torch checkpoint look truncated.
            b'.' => {
                s.end = i;
                return s;
            }

            // --- opcodes with no argument bytes ---
            b'(' | b'0' | b'1' | b'2' | b'N' | b'Q' | b'a' | b'd' | b'}' | b'e' | b'l' | b']'
            | b's' | b't' | b')' | b'u' | b'\x85' | b'\x86' | b'\x87' | b'\x88' | b'\x89'
            | b'\x8f' | b'\x90' | b'\x97' | b'\x98' | b'\x94' => {
                // MEMOIZE (\x94) memoizes top of stack; harmless to ignore here.
            }

            // --- execution / construction triggers ---
            b'R' => s.code_exec_ops.push("REDUCE"),
            b'b' => s.code_exec_ops.push("BUILD"),
            b'o' => s.code_exec_ops.push("OBJ"),
            b'\x81' => s.code_exec_ops.push("NEWOBJ"),
            b'\x92' => s.code_exec_ops.push("NEWOBJ_EX"),

            // --- newline-terminated argument(s) ---
            b'F' | b'I' | b'L' | b'P' | b'g' => {
                let _ = read_line(data, &mut i, &mut s);
            }
            b'S' | b'V' => {
                // STRING / UNICODE: one newline-terminated (quoted) value.
                if let Some(v) = read_line(data, &mut i, &mut s) {
                    str_stack.push(unquote(&v));
                }
            }
            b'c' => {
                // GLOBAL: module\n name\n
                let module = read_line(data, &mut i, &mut s);
                let name = read_line(data, &mut i, &mut s);
                if let (Some(m), Some(nm)) = (module, name) {
                    s.globals.push(GlobalRef {
                        opcode: "GLOBAL",
                        module: m,
                        name: nm,
                    });
                }
            }
            b'i' => {
                // INST: module\n name\n  (instantiates -> can run code)
                let module = read_line(data, &mut i, &mut s);
                let name = read_line(data, &mut i, &mut s);
                s.code_exec_ops.push("INST");
                if let (Some(m), Some(nm)) = (module, name) {
                    s.globals.push(GlobalRef {
                        opcode: "INST",
                        module: m,
                        name: nm,
                    });
                }
            }

            // --- fixed-size integer args ---
            b'K' | b'h' | b'q' | b'\x80' | b'\x82' => skip!(1),
            b'M' | b'\x83' => skip!(2),
            b'J' | b'j' | b'r' | b'\x84' => skip!(4),
            b'G' => skip!(8),

            // --- length-prefixed byte/str blobs ---
            b'U' | b'C' => {
                // 1-byte length
                need!(1);
                let len = data[i] as usize;
                i += 1;
                need!(len);
                let val = &data[i..i + len];
                i += len;
                str_stack.push(String::from_utf8_lossy(val).into_owned());
            }
            b'\x8c' => {
                // SHORT_BINUNICODE: 1-byte length
                need!(1);
                let len = data[i] as usize;
                i += 1;
                need!(len);
                str_stack.push(String::from_utf8_lossy(&data[i..i + len]).into_owned());
                i += len;
            }
            b'T' | b'X' | b'B' => {
                // 4-byte length
                need!(4);
                let len = u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
                i += 4;
                need!(len);
                str_stack.push(String::from_utf8_lossy(&data[i..i + len]).into_owned());
                i += len;
            }
            b'\x8d' | b'\x8e' | b'\x96' => {
                // BINUNICODE8 / BINBYTES8 / BYTEARRAY8: 8-byte length
                need!(8);
                let len = u64::from_le_bytes(data[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                need!(len);
                if op == b'\x8d' {
                    str_stack.push(String::from_utf8_lossy(&data[i..i + len]).into_owned());
                }
                i += len;
            }
            b'\x8a' => {
                // LONG1: 1-byte length + little-endian data.
                need!(1);
                let len = data[i] as usize;
                i += 1;
                need!(len);
                if s.first_long1.is_none() && len <= 16 {
                    let mut v: u128 = 0;
                    for (k, b) in data[i..i + len].iter().enumerate() {
                        v |= (*b as u128) << (8 * k);
                    }
                    s.first_long1 = Some(v);
                }
                i += len;
            }
            b'\x8b' => {
                // LONG4: 4-byte length + data
                need!(4);
                let len = u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
                i += 4;
                need!(len);
                i += len;
            }
            b'\x95' => {
                // FRAME: 8-byte frame length (informational; data follows normally)
                need!(8);
                i += 8;
            }
            b'\x93' => {
                // STACK_GLOBAL: pops name then module from the stack.
                let name = str_stack.pop();
                let module = str_stack.pop();
                if let (Some(m), Some(nm)) = (module, name) {
                    s.globals.push(GlobalRef {
                        opcode: "STACK_GLOBAL",
                        module: m,
                        name: nm,
                    });
                }
            }

            _ => {
                // Unknown / unsupported opcode: stop and flag as incomplete
                // rather than risk misinterpreting the rest of the stream.
                s.truncated = true;
                s.end = i - 1;
                return s;
            }
        }
    }

    // Ran out of bytes without ever reaching STOP.
    s.truncated = true;
    s.end = n;
    s
}

/// Split a raw pickle artifact into its streams and whatever follows them.
///
/// A legacy `torch.save` file is not one pickle: it is five concatenated
/// pickles (magic, protocol version, sys_info, the module, the storage keys)
/// followed by the raw tensor storages. Reading it as a single opcode stream
/// means dying on the first storage byte and calling a perfectly normal
/// checkpoint truncated.
fn split_streams(data: &[u8]) -> Container {
    let mut c = Container::default();
    let mut pos = 0usize;

    while pos < data.len() && c.streams.len() < MAX_STREAMS {
        let mut scan = scan_stream(&data[pos..]);
        let consumed = scan.end;
        let truncated = scan.truncated;
        if c.streams.is_empty() {
            c.torch_legacy = scan.first_long1 == Some(TORCH_LEGACY_MAGIC);
        }
        scan.end = pos + consumed;
        let stopped_at = scan.end;
        c.streams.push(scan);
        pos = stopped_at;
        c.stream_count = c.streams.len();
        c.analyzed_to = pos;

        if truncated {
            c.truncated_at = Some(pos);
            return c;
        }
        // Another stream follows only if it announces itself with PROTO.
        if pos >= data.len() || data[pos] != 0x80 {
            break;
        }
    }

    if pos < data.len() {
        c.tail = Some((pos, data.len() - pos));
    }
    c
}

/// Read a newline-terminated line, advancing the cursor past the `\n`.
fn read_line(data: &[u8], i: &mut usize, s: &mut StreamScan) -> Option<String> {
    let start = *i;
    while *i < data.len() && data[*i] != b'\n' {
        *i += 1;
    }
    if *i >= data.len() {
        s.truncated = true;
        return None;
    }
    let line = String::from_utf8_lossy(&data[start..*i]).into_owned();
    *i += 1; // skip '\n'
    Some(line)
}

/// Strip surrounding quotes from a STRING/UNICODE literal (best effort).
fn unquote(v: &str) -> String {
    let t = v.trim();
    let bytes = t.as_bytes();
    if bytes.len() >= 2
        && (bytes[0] == b'\'' || bytes[0] == b'"')
        && bytes[bytes.len() - 1] == bytes[0]
    {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the pickle a malicious `__reduce__` would emit for
    /// `os.system("...")` using protocol-2-style GLOBAL + REDUCE.
    fn os_system_pickle() -> Vec<u8> {
        let mut p = Vec::new();
        p.push(0x80); // PROTO
        p.push(0x02);
        p.extend_from_slice(b"cos\nsystem\n"); // GLOBAL os system
                                               // short binstring arg
        p.push(b'U');
        let arg = b"echo hi";
        p.push(arg.len() as u8);
        p.extend_from_slice(arg);
        p.push(b'\x85'); // TUPLE1
        p.push(b'R'); // REDUCE
        p.push(b'.'); // STOP
        p
    }

    #[test]
    fn detects_os_system_rce() {
        let report = analyze("evil.pkl", &os_system_pickle());
        assert_eq!(report.verdict, Verdict::Untrusted);
        let rce = report
            .findings
            .iter()
            .find(|f| f.id == "PICKLE_RCE_RISK")
            .expect("expected PICKLE_RCE_RISK");
        assert_eq!(rce.severity, Severity::High);
        assert!(
            rce.evidence.iter().any(|e| e.contains("os.system")),
            "evidence should name os.system, got {:?}",
            rce.evidence
        );
    }

    #[test]
    fn stack_global_resolves() {
        // proto 4 style: SHORT_BINUNICODE os, SHORT_BINUNICODE system, STACK_GLOBAL, REDUCE
        let mut p = vec![0x80, 0x04];
        for word in [b"os".as_slice(), b"system".as_slice()] {
            p.push(0x8c);
            p.push(word.len() as u8);
            p.extend_from_slice(word);
        }
        p.push(0x93); // STACK_GLOBAL
        p.push(b'.');
        let report = analyze("evil.pkl", &p);
        let rce = report.findings.iter().find(|f| f.id == "PICKLE_RCE_RISK");
        assert!(rce.is_some(), "STACK_GLOBAL os.system should be flagged");
    }
    // -----------------------------------------------------------------
    // Container layout: a legacy torch file is five pickles, then payload
    // -----------------------------------------------------------------

    /// `0x1950a86a20f9469cfc6c`, little-endian, as LONG1 encodes it.
    const TORCH_MAGIC_LE: [u8; 10] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19];

    /// The real layout `torch.save` writes without zip serialization: magic,
    /// protocol version, sys_info, the module, the storage keys, then raw
    /// tensor bytes.
    fn legacy_torch_container(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x80, 0x02, 0x8a, 0x0a]);
        out.extend_from_slice(&TORCH_MAGIC_LE);
        out.push(b'.');
        out.extend_from_slice(&[0x80, 0x02, 0x4d, 0xe9, 0x03, b'.']); // BININT2 1001
        out.extend_from_slice(&[0x80, 0x02, b'}', b'.']); // EMPTY_DICT
        out.extend_from_slice(&os_system_pickle_v2()); // the module pickle
        out.extend_from_slice(&[0x80, 0x02, b']', b'.']); // EMPTY_LIST
        out.extend_from_slice(payload);
        out
    }

    fn os_system_pickle_v2() -> Vec<u8> {
        let mut p = vec![0x80, 0x02];
        p.extend_from_slice(b"cos\nsystem\n");
        p.push(b'U');
        let arg = b"echo hi";
        p.push(arg.len() as u8);
        p.extend_from_slice(arg);
        p.push(b'\x85');
        p.push(b'R');
        p.push(b'.');
        p
    }

    #[test]
    fn stop_ends_a_stream_instead_of_running_into_the_next_byte() {
        let mut buf = os_system_pickle_v2();
        let opcodes = buf.len();
        buf.extend_from_slice(&[0xff; 16]); // bytes no opcode table can decode
        let scan = scan_stream(&buf);
        assert!(!scan.truncated, "STOP is not a truncation");
        assert_eq!(scan.end, opcodes);
    }

    #[test]
    fn a_legacy_container_is_split_into_its_streams() {
        let payload = vec![0x42u8; 4096];
        let data = legacy_torch_container(&payload);
        let c = split_streams(&data);

        assert!(c.torch_legacy, "the magic number identifies the container");
        assert_eq!(c.stream_count, 5, "magic, version, sys_info, module, keys");
        assert_eq!(c.truncated_at, None, "nothing was truncated");
        let (start, len) = c.tail.expect("the storage payload");
        assert_eq!(len, payload.len());
        assert_eq!(start, data.len() - payload.len());
    }

    /// The B6 regression: this used to report PICKLE_TRUNCATED on every
    /// ordinary legacy checkpoint, so nobody could tell a real one from noise.
    #[test]
    fn a_legacy_checkpoint_is_not_reported_as_truncated() {
        let data = legacy_torch_container(&[0x42u8; 4096]);
        let r = analyze("pytorch_model.bin", &data);
        let ids: Vec<&str> = r.findings.iter().map(|f| f.id.as_str()).collect();
        assert!(!ids.contains(&"PICKLE_TRUNCATED"), "{ids:?}");
        assert!(ids.contains(&"PICKLE_TORCH_LEGACY"), "{ids:?}");
        assert!(
            ids.contains(&"PICKLE_RCE_RISK"),
            "the payload is still caught"
        );

        let note = r
            .findings
            .iter()
            .find(|f| f.id == "PICKLE_TORCH_LEGACY")
            .unwrap();
        assert_eq!(note.severity, Severity::Info);
        assert!(
            note.detail.contains("5 pickle stream(s)"),
            "{}",
            note.detail
        );
        assert!(note.detail.contains("4096 byte(s)"), "{}", note.detail);
    }

    #[test]
    fn the_report_says_which_stream_carried_the_payload() {
        let data = legacy_torch_container(&[0u8; 32]);
        let r = analyze("pytorch_model.bin", &data);
        let rce = r
            .findings
            .iter()
            .find(|f| f.id == "PICKLE_RCE_RISK")
            .unwrap();
        assert!(
            rce.evidence.iter().any(|e| e.starts_with("stream 4/5")),
            "{:?}",
            rce.evidence
        );
    }

    #[test]
    fn a_genuinely_truncated_stream_is_still_reported_with_its_offset() {
        let full = os_system_pickle_v2();
        let cut = &full[..full.len() - 3];
        let r = analyze("broken.pkl", cut);
        let f = r
            .findings
            .iter()
            .find(|f| f.id == "PICKLE_TRUNCATED")
            .expect("truncation is still detected");
        assert!(f.detail.contains("stopped at byte"), "{}", f.detail);
    }

    #[test]
    fn data_stapled_after_a_pickle_is_flagged_by_signature() {
        let mut data = os_system_pickle_v2();
        let start = data.len();
        data.extend_from_slice(b"PK\x03\x04");
        data.extend_from_slice(&[0x41; 256]);

        let r = analyze("model.pkl", &data);
        let f = r
            .findings
            .iter()
            .find(|f| f.id == "PICKLE_TRAILING_DATA")
            .expect("trailing data");
        assert_eq!(f.severity, Severity::High);
        assert!(f.detail.contains("a ZIP archive"), "{}", f.detail);
        assert!(f.detail.contains(&start.to_string()), "{}", f.detail);
    }

    #[test]
    fn unexplained_trailing_bytes_are_flagged_lower_without_a_signature() {
        let mut data = os_system_pickle_v2();
        data.extend_from_slice(&[1u8, 2, 3].repeat(50));
        let r = analyze("model.pkl", &data);
        let f = r
            .findings
            .iter()
            .find(|f| f.id == "PICKLE_TRAILING_DATA")
            .unwrap();
        assert_eq!(f.severity, Severity::Medium);
    }

    #[test]
    fn an_ordinary_single_pickle_gets_no_container_finding() {
        let r = analyze("model.pkl", &os_system_pickle_v2());
        let ids: Vec<&str> = r.findings.iter().map(|f| f.id.as_str()).collect();
        assert!(!ids.contains(&"PICKLE_TRAILING_DATA"), "{ids:?}");
        assert!(!ids.contains(&"PICKLE_TORCH_LEGACY"), "{ids:?}");
        assert!(!ids.contains(&"PICKLE_TRUNCATED"), "{ids:?}");
    }
}
