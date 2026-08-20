//! Static analysis of Jinja chat templates.
//!
//! A chat template is not inert text: `transformers` renders it for every
//! conversation, so it runs before the model produces a single token. It is
//! rendered in a Jinja sandbox, which means a payload cannot simply call
//! `os.system`: it has to *escape* first, by pivoting through Python's object
//! graph (`__class__`, `__mro__`, `__subclasses__`, `__globals__`) or through
//! one of the Jinja globals that leak a reference to it (`lipsum`, `cycler`,
//! `joiner`).
//!
//! That is what makes the check worth having. Formatting a conversation needs
//! none of those. A template that reaches for them is not doing its job, and
//! saying so is a great deal more useful than "a template is present, go look".

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::hash;
use crate::report::{ArtifactReport, Finding, Severity, Verdict};

/// Attribute pivots that only appear when someone is walking the Python object
/// graph. None of them has a legitimate use in a chat template.
const ESCAPE_GADGETS: &[(&str, &str)] = &[
    ("__class__", "pivots from a value to its class"),
    ("__mro__", "walks the class hierarchy"),
    ("__bases__", "walks the class hierarchy"),
    ("__base__", "walks the class hierarchy"),
    ("__subclasses__", "enumerates every loaded class"),
    ("__globals__", "reaches a function's module globals"),
    ("__builtins__", "reaches the builtins namespace"),
    ("__import__", "imports a module by name"),
    ("__init__", "pivots through a constructor"),
    ("__reduce__", "reaches the pickle reduction protocol"),
    ("__getattribute__", "bypasses attribute lookup"),
    ("__dict__", "reads an object's raw attribute table"),
    ("func_globals", "reaches a function's module globals"),
    (
        "_TemplateReference__context",
        "reaches the template context",
    ),
];

/// Jinja globals that exist only to be used as a pivot in an escape chain.
/// `namespace` is deliberately absent: real chat templates use it.
const GADGET_GLOBALS: &[&str] = &["lipsum", "cycler", "joiner"];

/// Ways to name an attribute without writing it down, which is how a payload
/// gets past a filter that looks for `__class__`.
const DYNAMIC_ATTRIBUTE: &[&str] = &["|attr", "| attr", "attr(", "getattr("];

/// Cap per template, so a generated monster cannot flood the report.
const MAX_FINDINGS: usize = 12;

/// Files a repo can put a chat template in.
const TEMPLATE_FILES: [&str; 2] = ["chat_template.jinja", "chat_template.json"];

/// One `{{ … }}` or `{% … %}` region: the only places a template executes
/// anything. Literal text between them is just output.
struct Region {
    line: usize,
    text: String,
}

fn regions(src: &str) -> Vec<Region> {
    let b = src.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;

    while i < n {
        if b[i] == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if b[i] == b'{' && i + 1 < n && matches!(b[i + 1], b'{' | b'%' | b'#') {
            let open = b[i + 1];
            let close: &[u8] = match open {
                b'{' => b"}}",
                b'%' => b"%}",
                _ => b"#}",
            };
            let start = i;
            let start_line = line;
            let mut j = i + 2;
            while j + 1 < n && &b[j..j + 2] != close {
                if b[j] == b'\n' {
                    line += 1;
                }
                j += 1;
            }
            let end = (j + 2).min(n);
            // `{# … #}` is a comment: it renders nothing and runs nothing.
            if open != b'#' {
                out.push(Region {
                    line: start_line,
                    text: src[start..end].to_string(),
                });
            }
            i = end;
            continue;
        }
        i += 1;
    }
    out
}

/// Does `text` contain `needle` as something other than a fragment of a longer
/// identifier? Keeps `attr(` from matching `getattr(` twice over.
fn contains_token(text: &str, needle: &str) -> bool {
    let bytes = text.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(needle) {
        let at = from + rel;
        let prev_ok = at == 0 || {
            let p = bytes[at - 1];
            !(p.is_ascii_alphanumeric() || p == b'_')
        };
        if prev_ok {
            return true;
        }
        from = at + 1;
    }
    false
}

/// Signs that an attribute name is being assembled rather than written.
fn obfuscation(text: &str) -> Option<&'static str> {
    if text.contains("chr(") {
        return Some("character codes assembled into a name");
    }
    if text.matches("\\x").count() >= 4 {
        return Some("hex escapes in place of a name");
    }
    // `'__cl' + 'ass__'` or `'__cl' ~ 'ass__'`: a split dunder is never typing
    // convenience.
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    for sep in ["'+'", "'~'", "\"+\"", "\"~\""] {
        if compact.contains(sep) && compact.contains("__") {
            return Some("a dunder name split across concatenated strings");
        }
    }
    if compact.contains("|join") && compact.contains("__") {
        return Some("a dunder name assembled with the join filter");
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    let mut out: String = flat.chars().take(max).collect();
    out.push('…');
    out
}

/// Analyze one template's source. `origin` names it in the report.
pub fn analyze(source: &str, origin: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    for region in regions(source) {
        if findings.len() >= MAX_FINDINGS {
            break;
        }
        let text = &region.text;
        let line = region.line;
        let evidence = vec![format!("{origin} line {line}: {}", truncate(text, 140))];

        let gadgets: Vec<&(&str, &str)> = ESCAPE_GADGETS
            .iter()
            .filter(|(name, _)| contains_token(text, name))
            .collect();

        if !gadgets.is_empty() {
            let names: Vec<String> = gadgets
                .iter()
                .map(|(name, what)| format!("{name} ({what})"))
                .collect();
            findings.push(
                Finding::new(
                    "TEMPLATE_SANDBOX_ESCAPE",
                    Severity::High,
                    format!(
                        "chat template in {origin} reaches into Python internals: {}. \
                         Rendering happens on every conversation, so this runs before the \
                         model answers anything; formatting a chat needs none of it",
                        names.join(", ")
                    ),
                )
                .with_evidence(evidence.clone()),
            );
            continue; // one verdict per region is enough
        }

        let globals: Vec<&str> = GADGET_GLOBALS
            .iter()
            .copied()
            .filter(|g| contains_token(text, g))
            .collect();
        if !globals.is_empty() {
            findings.push(
                Finding::new(
                    "TEMPLATE_GADGET_GLOBAL",
                    Severity::Medium,
                    format!(
                        "chat template in {origin} uses the Jinja global(s) {}, which exist \
                         to generate filler text and are mainly known as sandbox-escape \
                         pivots; a chat template has no use for them",
                        globals.join(", ")
                    ),
                )
                .with_evidence(evidence.clone()),
            );
            continue;
        }

        let obfuscated = obfuscation(text);

        if let Some(dyn_attr) = DYNAMIC_ATTRIBUTE
            .iter()
            .find(|needle| text.contains(**needle))
        {
            // Looking an attribute up by computed name is suspicious. Doing it
            // with a name assembled from pieces is the documented way to write
            // `__class__` without writing `__class__`, and it is not an
            // accident anyone has by mistake.
            let (severity, extra) = match obfuscated {
                Some(kind) => (
                    Severity::High,
                    format!(", and the name is built at render time: {kind}"),
                ),
                None => (Severity::Medium, String::new()),
            };
            findings.push(
                Finding::new(
                    "TEMPLATE_DYNAMIC_ATTRIBUTE",
                    severity,
                    format!(
                        "chat template in {origin} looks up an attribute by computed name \
                         (`{}`), which is how a payload names `__class__` without writing \
                         it{extra}",
                        dyn_attr.trim()
                    ),
                )
                .with_evidence(evidence.clone()),
            );
            continue;
        }

        if let Some(kind) = obfuscated {
            findings.push(
                Finding::new(
                    "TEMPLATE_OBFUSCATION",
                    Severity::Medium,
                    format!("chat template in {origin} builds a name at render time: {kind}"),
                )
                .with_evidence(evidence),
            );
        }
    }

    findings
}

/// Find the chat templates a repo ships and analyze them.
///
/// Only templates with something to say produce a report. Every modern repo has
/// a chat template, and an artifact per repo saying "a template is present"
/// would be noise nobody reads.
pub fn scan_repo(root: &Path) -> Vec<ArtifactReport> {
    let base = if root.is_dir() {
        root.to_path_buf()
    } else {
        match root.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    };

    let mut reports = Vec::new();

    for name in TEMPLATE_FILES {
        let path = base.join(name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        // `chat_template.json` wraps the template in JSON.
        let source = if name.ends_with(".json") {
            json_templates(&text).join("\n")
        } else {
            text.clone()
        };
        let findings = analyze(&source, name);
        if !findings.is_empty() {
            reports.push(report_for(&path, "template", findings, text.as_bytes()));
        }
    }

    // Templates also live inside the tokenizer config.
    let path = base.join("tokenizer_config.json");
    if let Ok(text) = std::fs::read_to_string(&path) {
        let source = json_templates(&text).join("\n");
        if !source.is_empty() {
            let findings = analyze(&source, "tokenizer_config.json");
            if !findings.is_empty() {
                reports.push(report_for(&path, "config", findings, text.as_bytes()));
            }
        }
    }

    reports
}

fn report_for(path: &Path, format: &str, findings: Vec<Finding>, raw: &[u8]) -> ArtifactReport {
    let mut r = ArtifactReport::new(path.display().to_string(), format);
    r.hashes.file = Some(hash::tagged(&hash::blake3_hex(raw)));
    r.verdict = if findings.iter().any(|f| f.severity >= Severity::Medium) {
        Verdict::Untrusted
    } else {
        Verdict::Clean
    };
    for f in findings {
        r.push(f);
    }
    r
}

/// Pull every template string out of a JSON document, whether it is a bare
/// string or the list-of-named-templates form.
fn json_templates(text: &str) -> Vec<String> {
    let Ok(json) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut push = |v: &Value| match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(items) => {
            for it in items {
                if let Some(s) = it.get("template").and_then(|t| t.as_str()) {
                    out.push(s.to_string());
                }
            }
        }
        _ => {}
    };
    if let Some(v) = json.get("chat_template") {
        push(v);
    }
    // chat_template.json sometimes stores it at the top level.
    if json.get("chat_template").is_none() {
        push(&json);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(f: &[Finding]) -> Vec<&str> {
        f.iter().map(|x| x.id.as_str()).collect()
    }

    fn worst(f: &[Finding]) -> Option<Severity> {
        f.iter().map(|x| x.severity).max()
    }

    // -----------------------------------------------------------------
    // No false positives on the templates real models actually ship
    // -----------------------------------------------------------------

    /// Structurally faithful reproductions of the templates in wide use. If any
    /// of these ever produces a finding, the check is worse than useless: it
    /// would fire on nearly every modern repo and be ignored everywhere.
    const REAL_TEMPLATES: &[(&str, &str)] = &[
        (
            "llama-3 style",
            r#"{{- bos_token }}
{%- if custom_tools is defined %}{%- set tools = custom_tools %}{%- endif %}
{#- This block extracts the system message. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- system_message }}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '{"name": "' + tool_call.name + '", "parameters": ' }}
        {{- tool_call.arguments | tojson }}
    {%- endif %}
{%- endfor %}"#,
        ),
        (
            "chatml with tools",
            r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {{- "\n\n# Tools\n\n<tools>" }}
    {%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}
    {{- "\n</tools>" }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}"#,
        ),
        (
            "namespace bookkeeping",
            r#"{%- set ns = namespace(is_first=false, is_tool=false, system_prompt='') %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}{%- set ns.system_prompt = message['content'] %}{%- endif %}
{%- endfor %}
{{- bos_token }}{{- ns.system_prompt }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {%- set ns.is_tool = false %}
        {{- '<|User|>' + message['content'] }}
    {%- endif %}
    {%- if loop.last and add_generation_prompt %}{{- '<|Assistant|>' }}{%- endif %}
{%- endfor %}"#,
        ),
        (
            "mistral alternation",
            r#"{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('Conversation roles must alternate user/assistant!') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {{- '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] | trim + eos_token }}
    {%- endif %}
{%- endfor %}"#,
        ),
        (
            "string helpers and dates",
            r#"{%- set date_string = strftime_now("%d %b %Y") %}
{{- "Today: " + date_string }}
{%- for m in messages %}
    {{- m.content.strip() }}
    {%- if m.content.startswith("/") %}{{- " (command)" }}{%- endif %}
{%- endfor %}"#,
        ),
    ];

    #[test]
    fn real_chat_templates_produce_nothing() {
        for (name, src) in REAL_TEMPLATES {
            let f = analyze(src, "chat_template.jinja");
            assert!(f.is_empty(), "false positive on {name}: {:?}", ids(&f));
        }
    }

    #[test]
    fn namespace_is_not_treated_as_a_gadget() {
        // Real templates use it for bookkeeping; only `namespace.__init__` is
        // an escape, and that is caught by the dunder itself.
        let f = analyze("{%- set ns = namespace(x=1) %}{{ ns.x }}", "t");
        assert!(f.is_empty(), "{:?}", ids(&f));
    }

    #[test]
    fn literal_text_is_not_code() {
        // The words matter only inside {{ }} or {% %}. Documentation that
        // mentions __class__ is not an exploit.
        let f = analyze("Do not use __class__ or lipsum in templates.", "t");
        assert!(f.is_empty(), "{:?}", ids(&f));
    }

    #[test]
    fn a_comment_is_not_executed() {
        let f = analyze(
            "{# {{ ''.__class__.__mro__ }} #}{{ messages[0].content }}",
            "t",
        );
        assert!(f.is_empty(), "{:?}", ids(&f));
    }

    // -----------------------------------------------------------------
    // The escapes a sandboxed template needs to run anything
    // -----------------------------------------------------------------

    #[test]
    fn the_classic_mro_walk_is_caught() {
        let f = analyze("{{ ''.__class__.__mro__[1].__subclasses__() }}", "t");
        assert_eq!(ids(&f), vec!["TEMPLATE_SANDBOX_ESCAPE"]);
        assert_eq!(worst(&f), Some(Severity::High));
        assert!(f[0].detail.contains("__subclasses__"), "{}", f[0].detail);
    }

    #[test]
    fn gadget_globals_are_caught_with_and_without_dunders() {
        let with_dunder = analyze("{{ lipsum.__globals__['os'].popen('id').read() }}", "t");
        assert_eq!(ids(&with_dunder), vec!["TEMPLATE_SANDBOX_ESCAPE"]);

        // `cycler` alone has no business in a chat template either.
        let bare = analyze("{{ cycler }}", "t");
        assert_eq!(ids(&bare), vec!["TEMPLATE_GADGET_GLOBAL"]);
        assert_eq!(worst(&bare), Some(Severity::Medium));
    }

    #[test]
    fn the_template_context_pivot_is_caught() {
        let f = analyze(
            "{{ self._TemplateReference__context.cycler.__init__.__globals__.os }}",
            "t",
        );
        assert_eq!(ids(&f), vec!["TEMPLATE_SANDBOX_ESCAPE"]);
    }

    #[test]
    fn a_dunder_split_across_strings_is_not_a_hiding_place() {
        let f = analyze("{{ ''|attr('__cl'+'ass__') }}", "t");
        assert_eq!(worst(&f), Some(Severity::High), "{:?}", ids(&f));
    }

    #[test]
    fn computed_attribute_names_are_scored_by_how_hidden_they_are() {
        // Plain dynamic lookup: suspicious.
        let plain = analyze("{{ getattr(obj, name) }}", "t");
        assert_eq!(ids(&plain), vec!["TEMPLATE_DYNAMIC_ATTRIBUTE"]);
        assert_eq!(worst(&plain), Some(Severity::Medium));

        // Dynamic lookup with an assembled name: that is the exploit.
        let hidden = analyze("{% set c = getattr(ns, '__in'~'it__') %}", "t");
        assert_eq!(ids(&hidden), vec!["TEMPLATE_DYNAMIC_ATTRIBUTE"]);
        assert_eq!(worst(&hidden), Some(Severity::High));
    }

    #[test]
    fn character_arithmetic_is_reported() {
        let f = analyze("{{ x[chr(95)+chr(95)+'class'] }}", "t");
        assert_eq!(ids(&f), vec!["TEMPLATE_OBFUSCATION"]);
    }

    #[test]
    fn a_payload_hidden_late_in_a_long_template_is_still_found() {
        let mut src = String::from("{%- for m in messages %}{{ m.content }}{%- endfor %}\n");
        for _ in 0..200 {
            src.push_str("{{ m.role }}\n");
        }
        src.push_str("{% if 0 %}{{ ''.__class__ }}{% endif %}");
        let f = analyze(&src, "t");
        assert!(
            ids(&f).contains(&"TEMPLATE_SANDBOX_ESCAPE"),
            "{:?}",
            ids(&f)
        );
    }

    #[test]
    fn findings_are_capped() {
        let src = "{{ ''.__class__ }}\n".repeat(50);
        assert!(analyze(&src, "t").len() <= MAX_FINDINGS);
    }

    #[test]
    fn the_evidence_names_the_line() {
        let f = analyze("{{ messages }}\n{{ ''.__class__ }}", "chat_template.jinja");
        assert!(
            f[0].evidence[0].starts_with("chat_template.jinja line 2:"),
            "{:?}",
            f[0].evidence
        );
    }

    // -----------------------------------------------------------------
    // Where templates live
    // -----------------------------------------------------------------

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("assay-tpl-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_standalone_template_file_is_reported_as_its_own_artifact() {
        let dir = tmpdir("file");
        std::fs::write(
            dir.join("chat_template.jinja"),
            "{{ lipsum.__globals__['os'].popen('id').read() }}",
        )
        .unwrap();

        let reports = scan_repo(&dir);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].format, "template");
        assert_eq!(reports[0].verdict, Verdict::Untrusted);
        assert!(reports[0].hashes.file.is_some(), "and it is pinnable");
    }

    #[test]
    fn a_template_inside_the_tokenizer_config_is_found() {
        let dir = tmpdir("tokcfg");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"model_max_length": 4096,
                "chat_template": "{{ ''.__class__.__mro__[1].__subclasses__() }}"}"#,
        )
        .unwrap();

        let reports = scan_repo(&dir);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].format, "config");
        assert!(reports[0].findings[0].id == "TEMPLATE_SANDBOX_ESCAPE");
    }

    #[test]
    fn the_named_template_list_form_is_understood() {
        let dir = tmpdir("named");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": [{"name": "default", "template": "{{ messages }}"},
                                  {"name": "tool_use", "template": "{{ cycler.__init__ }}"}]}"#,
        )
        .unwrap();
        assert_eq!(scan_repo(&dir).len(), 1);
    }

    #[test]
    fn an_ordinary_repo_produces_no_template_report() {
        let dir = tmpdir("quiet");
        std::fs::write(dir.join("chat_template.jinja"), REAL_TEMPLATES[1].1).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"model_max_length": 4096}"#,
        )
        .unwrap();
        assert!(
            scan_repo(&dir).is_empty(),
            "a normal template must not add noise to every scan"
        );
    }
}
