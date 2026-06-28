//! Pickle / arbitrary-code-execution risk scanner — the highest-priority check.
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
    truncated: bool,
}

/// Analyze a pickle artifact (raw stream or torch zip container).
pub fn analyze(artifact_name: &str, data: &[u8]) -> ArtifactReport {
    let mut report = ArtifactReport::new(artifact_name, "pickle");
    // A pickle artifact can run code at load time by design — never clean.
    report.verdict = Verdict::Untrusted;

    let mut scans: Vec<(String, StreamScan)> = Vec::new();

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
        scans.push((String::new(), scan_stream(data)));
    }

    summarize(&mut report, &scans);
    report
}

fn summarize(report: &mut ArtifactReport, scans: &[(String, StreamScan)]) {
    let mut dangerous_evidence: Vec<String> = Vec::new();
    let mut exec_evidence: Vec<String> = Vec::new();
    let mut any_dangerous = false;
    let mut any_exec = false;
    let mut any_global = false;
    let mut any_truncated = false;

    for (entry, scan) in scans {
        let prefix = if entry.is_empty() {
            String::new()
        } else {
            format!("{entry}: ")
        };
        if scan.truncated {
            any_truncated = true;
        }
        if !scan.code_exec_ops.is_empty() {
            any_exec = true;
        }
        for g in &scan.globals {
            any_global = true;
            if g.is_dangerous() {
                any_dangerous = true;
                if dangerous_evidence.len() < EVIDENCE_CAP {
                    dangerous_evidence
                        .push(format!("{prefix}opcode {} -> {}", g.opcode, g.qualified()));
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
            exec_evidence.push(format!(
                "{prefix}execution opcodes: {}",
                seen.join(", ")
            ));
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

    if any_truncated {
        report.push(Finding::new(
            "PICKLE_TRUNCATED",
            Severity::Medium,
            "pickle opcode stream ended unexpectedly or hit an unknown opcode; \
             analysis may be incomplete",
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
            // --- opcodes with no argument bytes ---
            b'(' | b'.' | b'0' | b'1' | b'2' | b'N' | b'Q' | b'a' | b'd' | b'}' | b'e' | b'l'
            | b']' | b's' | b't' | b')' | b'u' | b'\x85' | b'\x86' | b'\x87' | b'\x88'
            | b'\x89' | b'\x8f' | b'\x90' | b'\x97' | b'\x98' | b'\x94' => {
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
                // LONG1: 1-byte length + data
                need!(1);
                let len = data[i] as usize;
                i += 1;
                need!(len);
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
                return s;
            }
        }
    }

    s
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
}
